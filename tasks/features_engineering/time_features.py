import pandas as pd
import numpy as np
import os
import logging
from typing import Optional
from datetime import datetime, timezone


def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'features_engineering', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'time_features.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'published_date' column.

    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    if 'published_date' not in df.columns:
        logger.error("'published_date' column not found")
        raise ValueError("'published_date' column not found")

    df = df.copy()
    df['published_date'] = pd.to_datetime(df['published_date'], utc=True, errors='coerce')
    
    # Initialize features
    features = [
        'day_of_week', 'hour', 'is_weekend', 'is_market_hours',
        'is_earnings_season', 'is_quarter_end', 'days_since_event'
    ]
    for feature in features:
        df[feature] = np.nan if feature != 'is_weekend' else 0

    # Add time-based features
    df['day_of_week'] = df['published_date'].dt.dayofweek
    df['hour'] = df['published_date'].dt.hour
    df['is_weekend'] = df['published_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Market hours (9:30 AM - 4:00 PM EST, assuming US markets)
    df['is_market_hours'] = ((df['published_date'].dt.hour >= 9) & 
                             (df['published_date'].dt.hour < 16)).astype(int)
    
    # Earnings season (approximated as 2 weeks after quarter end)
    df['month'] = df['published_date'].dt.month
    df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
    df['is_earnings_season'] = ((df['month'].isin([1, 4, 7, 10])) | 
                                (df['month'].isin([2, 5, 8, 11]))).astype(int)
    
    # Days since event
    current_time = datetime.now(timezone.utc)
    df['days_since_event'] = (current_time - df['published_date']).dt.days
    
    # Drop temporary column
    df = df.drop(columns=['month'], errors='ignore')
    
    # Log feature statistics
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats = df[features].describe().to_dict()
    stats_path = os.path.join(base_dir, 'data', 'features_reports', f'time_feature_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    pd.DataFrame(stats).to_csv(stats_path, index=True)
    logger.info(f"Saved time feature statistics to {stats_path}")
    
    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, 'data', 'feature_engineering', f'yfinance_enriched_data_{datetime.now().strftime("%Y%m%d")}.csv')
    df = pd.read_csv(input_path)
    df_with_time_features = add_time_features(df)
    df_with_time_features.to_csv(os.path.join(base_dir, 'data', 'feature_engineering', f'time_features_data_{datetime.now().strftime("%Y%m%d")}.csv'), index=False)

