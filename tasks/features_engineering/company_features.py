import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import yfinance as yf
import pickle
from textblob import TextBlob

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'features_engineering', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'company_features.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)


def validate_ticker(ticker: str) -> bool:
    """Validate ticker by checking if it has valid price data."""
    try:
        yf_ticker = yf.Ticker(ticker)
        # Attempt to fetch a small amount of data to verify ticker
        hist = yf_ticker.history(period="1d")
        if hist.empty or hist['Close'].isna().all():
            logger.warning(f"Ticker {ticker} is invalid or has no price data")
            return False
        return True
    except Exception as e:
        logger.warning(f"Ticker {ticker} validation failed: {str(e)}")
        return False

def calculate_volatility(ticker: str, period: str = "1y") -> float:
    """Calculate annualized volatility from 1-year historical data."""
    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period=period)
        if hist.empty or len(hist) < 10 or hist['Close'].isna().any():
            logger.warning(f"No valid price data for {ticker}")
            return np.nan
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        return volatility
    except Exception as e:
        logger.warning(f"Failed to calculate volatility for {ticker}: {str(e)}")
        return np.nan

def calculate_sector_volatility(sector: str, period: str = "1y") -> float:
    """Calculate sector volatility using sector indices."""
    sector_indices = {
        'Technology': '^IXIC',
        'Finance': '^IXBK',
        'Healthcare': '^IXHC',
        'Default': '^GSPC'
    }
    sector_ticker = sector_indices.get(sector, sector_indices['Default'])
    return calculate_volatility(sector_ticker, period)

def add_company_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add company-specific features to the DataFrame.

    Args:
        df: Input DataFrame with columns including yf_ticker, market_cap, sector, title, content

    Returns:
        pd.DataFrame: DataFrame with added features
    """
    logger.info("Starting company features addition")
    
    # Validate input columns
    required_columns = ['yf_ticker', 'market_cap', 'sector', 'title', 'content']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    # Step 1: Calculate combined sentiment using TextBlob
    logger.info("Calculating combined_sentiment using TextBlob")
    def get_sentiment(text: str) -> float:
        if pd.isna(text) or not isinstance(text, str):
            return 0.0
        return TextBlob(text).sentiment.polarity

    df['combined_sentiment'] = df.apply(
        lambda row: get_sentiment(str(row['title']) + ' ' + str(row['content'])), axis=1
    )

    # Step 2: Add market cap category
    logger.info("Adding market cap category feature")
    def categorize_market_cap(market_cap: float) -> str:
        if pd.isna(market_cap):
            return 'Unknown'
        if market_cap < 2e9:
            return 'Small'
        elif market_cap <= 10e9:
            return 'Mid'
        else:
            return 'Large'
    
    df['market_cap_category'] = df['market_cap'].apply(categorize_market_cap)

    # Step 3: Calculate volatility and sector-relative volatility
    logger.info("Calculating volatility and sector-relative volatility")
    invalid_tickers = []
    df['volatility'] = np.nan
    df['sector_relative_volatility'] = np.nan
    
    for idx, row in df.iterrows():
        ticker = row['yf_ticker']
        sector = row['sector']
        
        if not validate_ticker(ticker):
            invalid_tickers.append(ticker)
            continue
        
        # Calculate stock volatility
        df.at[idx, 'volatility'] = calculate_volatility(ticker)
        
        # Calculate sector volatility
        sector_vol = calculate_sector_volatility(sector)
        if not pd.isna(df.at[idx, 'volatility']) and not pd.isna(sector_vol) and sector_vol != 0:
            df.at[idx, 'sector_relative_volatility'] = df.at[idx, 'volatility'] / sector_vol

    # Impute missing volatility values
    if invalid_tickers:
        logger.info(f"Invalid tickers found: {invalid_tickers}")
        median_vol = df['volatility'].median()
        median_sector_rel_vol = df['sector_relative_volatility'].median()
        df['volatility'] = df['volatility'].fillna(median_vol)
        df['sector_relative_volatility'] = df['sector_relative_volatility'].fillna(median_sector_rel_vol)
        logger.info(f"Imputed {len(invalid_tickers)} missing volatility values with median: {median_vol}")
        logger.info(f"Imputed {len(invalid_tickers)} missing sector_relative_volatility values with median: {median_sector_rel_vol}")

    # Step 4: Calculate previous news sentiment
    logger.info("Calculating previous news sentiment")
    df['prev_news_sentiment'] = df.groupby('yf_ticker')['combined_sentiment'].shift(1).fillna(0.0)

    # Save feature statistics
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats_path = os.path.join(base_dir, 'data', 'features_reports', f'company_features_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats = df[['combined_sentiment', 'market_cap_category', 'volatility', 
                'sector_relative_volatility', 'prev_news_sentiment']].describe()
    stats.to_csv(stats_path)
    logger.info(f"Saved feature statistics to {stats_path}")

    logger.info("Company features addition completed")
    return df

if __name__ == '__main__':
    input_path = r"data\feature_engineering\time_features_data_20250707.csv"
    output_path = r"data\feature_engineering\company_features_data_20250707.csv"
    
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
        df = add_company_features(df)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved company features data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to process company features: {str(e)}")
        raise


