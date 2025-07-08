import sys
import os

from tasks.features_engineering.company_features import add_company_features
from tasks.features_engineering.features_selection import select_features
from tasks.features_engineering.fetch_yfinance_data import enrich_with_yfinance
from tasks.features_engineering.time_features import add_time_features

import pandas as pd
from datetime import datetime
import logging


def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'features_engineering', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'feature_engineering_pipeline.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

def run_feature_engineering_pipeline(
    input_path: str = r"data\clean\validated_price_moves_20250707.csv",
    yfinance_output_path: str = r"data\feature_engineering\yfinance_enriched_data_20250707.csv",
    time_features_output_path: str = r"data\feature_engineering\time_features_data_20250707.csv",
    company_features_output_path: str = r"data\feature_engineering\company_features_data_20250707.csv",
    final_output_path: str = r"feature_engineering\selected_features_data_20250707.csv"
) -> pd.DataFrame:
    """
    Run the feature engineering pipeline.

    Args:
        input_path: Path to validated input CSV
        yfinance_output_path: Path to save Yahoo Finance enriched data
        time_features_output_path: Path to save time features data
        company_features_output_path: Path to save company features data
        final_output_path: Path to save final enriched CSV

    Returns:
        pd.DataFrame: Enriched DataFrame with selected features
    """
    logger.info("Starting feature engineering pipeline")
    
    # Load validated data
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load {input_path}: {str(e)}")
        raise FileNotFoundError(f"Input file {input_path} not found. Ensure data quality pipeline ran successfully.")
    except Exception as e:
        logger.error(f"Failed to load {input_path}: {str(e)}")
        raise

    # Step 1: Enrich with Yahoo Finance data
    try:
        df = enrich_with_yfinance(df)
        os.makedirs(os.path.dirname(yfinance_output_path), exist_ok=True)
        df.to_csv(yfinance_output_path, index=False)
        logger.info(f"Saved Yahoo Finance enriched data to {yfinance_output_path}")
    except Exception as e:
        logger.error(f"Yahoo Finance enrichment failed: {str(e)}")
        raise ValueError(f"Yahoo Finance enrichment failed: {str(e)}")

    # Step 2: Add time-based features
    try:
        df = add_time_features(df)
        os.makedirs(os.path.dirname(time_features_output_path), exist_ok=True)
        df.to_csv(time_features_output_path, index=False)
        logger.info(f"Saved time features data to {time_features_output_path}")
    except Exception as e:
        logger.error(f"Time features addition failed: {str(e)}")
        raise ValueError(f"Time features addition failed: {str(e)}")

    # Step 3: Add company-specific features
    try:
        df = add_company_features(df)
        os.makedirs(os.path.dirname(company_features_output_path), exist_ok=True)
        df.to_csv(company_features_output_path, index=False)
        logger.info(f"Saved company features data to {company_features_output_path}")
    except Exception as e:
        logger.error(f"Company features addition failed: {str(e)}")
        raise ValueError(f"Company features addition failed: {str(e)}")

    # Step 4: Perform feature selection
    try:
        params = {
            'method': 'correlation_and_rf',
            'correlation_threshold': 0.8,
            'importance_threshold': 0.01
        }
        df, selected_features = select_features(df, params)
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise ValueError(f"Feature selection failed: {str(e)}")

    # Save final output
    try:
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        df.to_csv(final_output_path, index=False)
        logger.info(f"Saved final enriched data to {final_output_path} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to save {final_output_path}: {str(e)}")
        raise ValueError(f"Failed to save {final_output_path}: {str(e)}")

    logger.info("Feature engineering pipeline completed")
    return df

if __name__ == '__main__':
    df = run_feature_engineering_pipeline()
    logger.info(f"Pipeline completed. Final data saved to {r'data\feature_engineering\selected_features_data_20250707.csv'}")

