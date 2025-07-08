import numpy as np
import pandas as pd
import yfinance as yf
import os
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pickle
from retrying import retry

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'features_engineering', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'yfinance.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

def retry_if_exception(exception):
    """Retry on any exception."""
    return True

@retry(stop_max_attempt_number=3, wait_fixed=2000, retry_on_exception=retry_if_exception)
def fetch_yf_data(ticker: str, date: datetime) -> Dict:
    """Fetch Yahoo Finance data with retry."""
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info
    history = yf_ticker.history(period='1mo', end=date)
    return info, history

def validate_ticker(ticker: str) -> bool:
    """Validate if a ticker is supported by Yahoo Finance."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        return bool(info.get('symbol'))
    except Exception as e:
        logger.warning(f"Ticker {ticker} validation failed: {str(e)}")
        return False

def enrich_with_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich DataFrame with Yahoo Finance data.

    Args:
        df (pd.DataFrame): Input DataFrame with 'yf_ticker' and 'published_date' columns.

    Returns:
        pd.DataFrame: Enriched DataFrame with Yahoo Finance features.
    """
    if 'yf_ticker' not in df.columns or 'published_date' not in df.columns:
        logger.error("Required columns 'yf_ticker' and 'published_date' not found")
        raise ValueError("Required columns 'yf_ticker' and 'published_date' not found")

    df = df.copy()
    metrics = {'successful_fetches': 0, 'failed_fetches': 0, 'invalid_tickers': 0}
    base_dir = r"C:\Users\HP\Desktop\Upwork_project\finespresso-modelling"
    cache_path = os.path.join(base_dir, 'data', 'cache', 'yfinance_cache.pkl')
    cache = load_cache(cache_path)

    features = [
        'market_cap', 'float_shares', 'exchange', 'sector', 'industry',
        'avg_volume', 'beta', 'recent_volume', 'float_ratio', 'sector_performance'
    ]
    for feature in features:
        if feature in ['exchange', 'sector', 'industry']:
            df[feature] = ''
            df[feature] = df[feature].astype('object')
        else:
            df[feature] = np.nan

    sector_indices = {
        'Technology': '^IXIC',
        'Finance': '^IXBK',
        'Healthcare': '^IXHC',
        'default': '^GSPC'
    }

    for idx, row in df.iterrows():
        ticker = row['yf_ticker']
        date = pd.to_datetime(row['published_date']).tz_localize(None)
        cache_key = f"{ticker}_{date.strftime('%Y%m%d')}"

        if not validate_ticker(ticker):
            metrics['invalid_tickers'] += 1
            logger.warning(f"Skipping invalid ticker {ticker}")
            cache[cache_key] = {f: '' if f in ['exchange', 'sector', 'industry'] else np.nan for f in features}
            df.loc[idx, features] = pd.Series(cache[cache_key])
            continue

        if cache_key in cache and isinstance(cache[cache_key], dict) and all(f in cache[cache_key] for f in features):
            try:
                df.loc[idx, features] = pd.Series(cache[cache_key])
                logger.info(f"Used cached Yahoo Finance data for {ticker}")
                continue
            except Exception as e:
                logger.warning(f"Corrupted cache for {ticker}: {str(e)}. Refetching data.")
                del cache[cache_key]

        try:
            info, history = fetch_yf_data(ticker, date)
            if not info.get('symbol'):
                raise ValueError(f"No data returned for {ticker}")

            df.at[idx, 'market_cap'] = info.get('marketCap', np.nan)
            df.at[idx, 'float_shares'] = info.get('floatShares', np.nan)
            df.at[idx, 'exchange'] = info.get('exchange', '')
            df.at[idx, 'sector'] = info.get('sector', '')
            df.at[idx, 'industry'] = info.get('industry', '')
            df.at[idx, 'beta'] = info.get('beta', np.nan)

            if not history.empty:
                df.at[idx, 'avg_volume'] = history['Volume'].mean()
                df.at[idx, 'recent_volume'] = history['Volume'].iloc[-1] if len(history) > 0 else np.nan
            else:
                df.at[idx, 'avg_volume'] = np.nan
                df.at[idx, 'recent_volume'] = np.nan

            float_shares = df.at[idx, 'float_shares']
            market_cap = df.at[idx, 'market_cap']
            if pd.notna(float_shares) and pd.notna(market_cap) and market_cap != 0:
                df.at[idx, 'float_ratio'] = float_shares / market_cap
            else:
                df.at[idx, 'float_ratio'] = np.nan

            sector = info.get('sector', 'default')
            sector_index = sector_indices.get(sector, sector_indices['default'])
            sector_data = yf.Ticker(sector_index).history(period='1mo', end=date)
            if not sector_data.empty:
                df.at[idx, 'sector_performance'] = sector_data['Close'].pct_change().mean() * 100
            else:
                df.at[idx, 'sector_performance'] = np.nan

            cache[cache_key] = df.loc[idx, features].to_dict()
            metrics['successful_fetches'] += 1
            logger.info(f"Fetched Yahoo Finance data for {ticker}")
        except Exception as e:
            metrics['failed_fetches'] += 1
            logger.warning(f"Failed to fetch data for {ticker}: {str(e)}")
            cache[cache_key] = {f: '' if f in ['exchange', 'sector', 'industry'] else np.nan for f in features}
            df.loc[idx, features] = pd.Series(cache[cache_key])

    save_cache(cache, cache_path)
    metrics_path = os.path.join(base_dir, 'data', 'quality_metrics', f'yfinance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(f"Saved Yahoo Finance metrics to {metrics_path}")
    return df

def load_cache(cache_path: str) -> Dict:
    """Load Yahoo Finance cache."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {str(e)}")
            return {}
    return {}

def save_cache(cache: Dict, cache_path: str) -> None:
    """Save Yahoo Finance cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Saved Yahoo Finance cache to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save cache to {cache_path}: {str(e)}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, 'data', 'clean', f'validated_price_moves_{datetime.now().strftime("%Y%m%d")}.csv')
    df = pd.read_csv(input_path)
    enriched_df = enrich_with_yfinance(df)
    enriched_df.to_csv(os.path.join(base_dir, 'data', 'feature_engineering', f'yfinance_enriched_data_{datetime.now().strftime("%Y%m%d")}.csv'), index=False)

'''
import numpy as np
import pandas as pd
import yfinance as yf
import os
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pickle
from retrying import retry

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'data', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'yfinance.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

def retry_if_exception(exception):
    """Retry on any exception."""
    return True

@retry(stop_max_attempt_number=3, wait_fixed=2000, retry_on_exception=retry_if_exception)
def fetch_yf_data(ticker: str, date: datetime) -> Dict:
    """Fetch Yahoo Finance data with retry."""
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info
    history = yf_ticker.history(period='1mo', end=date)
    return info, history

def validate_ticker(ticker: str) -> bool:
    """Validate if a ticker is supported by Yahoo Finance."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        return bool(info.get('symbol'))
    except Exception as e:
        logger.warning(f"Ticker {ticker} validation failed: {str(e)}")
        return False

def enrich_with_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich DataFrame with Yahoo Finance data.

    Args:
        df (pd.DataFrame): Input DataFrame with 'ticker' and 'published_date' columns.

    Returns:
        pd.DataFrame: Enriched DataFrame with Yahoo Finance features.
    """
    if 'ticker' not in df.columns or 'published_date' not in df.columns:
        logger.error("Required columns 'ticker' and 'published_date' not found")
        raise ValueError("Required columns not found")

    df = df.copy()
    metrics = {'successful_fetches': 0, 'failed_fetches': 0, 'invalid_tickers': 0}
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_path = os.path.join(base_dir, 'data', 'cache', 'yfinance_cache.pkl')
    cache = load_cache(cache_path)

    features = [
        'market_cap', 'float_shares', 'exchange', 'sector', 'industry',
        'avg_volume', 'beta', 'recent_volume', 'float_ratio', 'sector_performance'
    ]
    # Initialize numerical columns with np.nan and categorical with empty strings
    for feature in features:
        if feature in ['exchange', 'sector', 'industry']:
            df[feature] = ''
            df[feature] = df[feature].astype('object')
        else:
            df[feature] = np.nan

    # Sector index mapping
    sector_indices = {
        'Technology': '^IXIC',  # NASDAQ for tech
        'Finance': '^IXBK',    # Banking index
        'Healthcare': '^IXHC',  # Healthcare index
        'default': '^GSPC'     # S&P 500 for others
    }

    for idx, row in df.iterrows():
        ticker = row['ticker']
        date = pd.to_datetime(row['published_date']).tz_localize(None)
        cache_key = f"{ticker}_{date.strftime('%Y%m%d')}"

        # Validate ticker
        if not validate_ticker(ticker):
            metrics['invalid_tickers'] += 1
            logger.warning(f"Skipping invalid ticker {ticker}")
            cache[cache_key] = {f: '' if f in ['exchange', 'sector', 'industry'] else np.nan for f in features}
            df.loc[idx, features] = pd.Series(cache[cache_key])
            continue

        # Check cache
        if cache_key in cache and isinstance(cache[cache_key], dict) and all(f in cache[cache_key] for f in features):
            try:
                df.loc[idx, features] = pd.Series(cache[cache_key])
                logger.info(f"Used cached Yahoo Finance data for {ticker}")
                continue
            except Exception as e:
                logger.warning(f"Corrupted cache for {ticker}: {str(e)}. Refetching data.")
                del cache[cache_key]  # Remove corrupted cache entry

        try:
            info, history = fetch_yf_data(ticker, date)
            if not info.get('symbol'):
                raise ValueError(f"No data returned for {ticker}")

            # Basic features
            df.at[idx, 'market_cap'] = info.get('marketCap', np.nan)
            df.at[idx, 'float_shares'] = info.get('floatShares', np.nan)
            df.at[idx, 'exchange'] = info.get('exchange', '')
            df.at[idx, 'sector'] = info.get('sector', '')
            df.at[idx, 'industry'] = info.get('industry', '')
            df.at[idx, 'beta'] = info.get('beta', np.nan)

            # Volume features
            if not history.empty:
                df.at[idx, 'avg_volume'] = history['Volume'].mean()
                df.at[idx, 'recent_volume'] = history['Volume'].iloc[-1] if len(history) > 0 else np.nan
            else:
                df.at[idx, 'avg_volume'] = np.nan
                df.at[idx, 'recent_volume'] = np.nan

            # Float ratio
            float_shares = df.at[idx, 'float_shares']
            market_cap = df.at[idx, 'market_cap']
            if pd.notna(float_shares) and pd.notna(market_cap) and market_cap != 0:
                df.at[idx, 'float_ratio'] = float_shares / market_cap
            else:
                df.at[idx, 'float_ratio'] = np.nan

                
            # Sector performance
            sector = info.get('sector', 'default')
            sector_index = sector_indices.get(sector, sector_indices['default'])
            sector_data = yf.Ticker(sector_index).history(period='1mo', end=date)
            if not sector_data.empty:
                df.at[idx, 'sector_performance'] = sector_data['Close'].pct_change().mean() * 100
            else:
                df.at[idx, 'sector_performance'] = np.nan

            cache[cache_key] = df.loc[idx, features].to_dict()
            metrics['successful_fetches'] += 1
            logger.info(f"Fetched Yahoo Finance data for {ticker}")
        except Exception as e:
            metrics['failed_fetches'] += 1
            logger.warning(f"Failed to fetch data for {ticker}: {str(e)}")
            cache[cache_key] = {f: '' if f in ['exchange', 'sector', 'industry'] else np.nan for f in features}
            df.loc[idx, features] = pd.Series(cache[cache_key])

    save_cache(cache, cache_path)
    metrics_path = os.path.join(base_dir, 'data', 'quality_metrics', f'yfinance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(f"Saved Yahoo Finance metrics to {metrics_path}")
    return df

def load_cache(cache_path: str) -> Dict:
    """Load Yahoo Finance cache."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {str(e)}")
            return {}
    return {}

def save_cache(cache: Dict, cache_path: str) -> None:
    """Save Yahoo Finance cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Saved Yahoo Finance cache to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save cache to {cache_path}: {str(e)}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = pd.read_csv(os.path.join(base_dir, 'data', 'clean', 'clean_price_moves.csv'))
    enriched_df = enrich_with_yfinance(df)
    enriched_df.to_csv(os.path.join(base_dir, 'data', 'feature_engineering', 'yfinance_enriched_data.csv'), index=False)
    '''