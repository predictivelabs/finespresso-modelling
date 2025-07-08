import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime
import logging
from typing import Dict
from tasks.data_cleaning.data_cleaner import DataCleaner
from tasks.data_cleaning.data_metrics import DataMetricsMonitor
from tasks.data_cleaning.data_validation import DataValidator
from tasks.data_cleaning.data_versioning import DataVersioning

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'data_cleaning', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'pipeline.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

def run_data_quality_pipeline(
    input_path: str = 'data/backfill_data/price_moves_backfill_20250706_151035.csv',
    cleaned_output_path: str = f'data/clean/cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv',
    validated_output_path: str = f'data/clean/validated_price_moves_{datetime.now().strftime("%Y%m%d")}.csv',
    metrics_dir: str = 'data/quality_metrics',
    versions_dir: str = 'data/versions',
    lineage_dir: str = 'data/lineage'
) -> Dict:
    """Run the full data quality pipeline for Step 1."""
    logger.info("Starting data quality pipeline")
    
    # Step 1: Clean Data
    logger.info("Running data cleaning")
    try:
        cleaner = DataCleaner(
            input_path=input_path,
            output_path=cleaned_output_path,
            metrics_path=os.path.join(metrics_dir, 'cleaning_metrics.csv')
        )
        cleaned_df = cleaner.clean()
        logger.info(f"Data cleaning completed. Output saved to {cleaned_output_path}")
    except FileNotFoundError as e:
        logger.error(f"Cleaning failed: Input file {input_path} not found")
        raise FileNotFoundError(f"Input file {input_path} not found. Ensure the backfill data exists.")
    except Exception as e:
        logger.error(f"Cleaning failed: {str(e)}")
        raise ValueError(f"Data cleaning failed: {str(e)}")
    
    # Step 2: Validate Data
    logger.info("Running data validation")
    try:
        validator = DataValidator(
            input_path=cleaned_output_path,
            output_path=validated_output_path,
            metrics_path=os.path.join(metrics_dir, 'validation_metrics.csv')
        )
        validated_df = validator.validate()
        logger.info(f"Data validation completed. Output saved to {validated_output_path}")
    except FileNotFoundError as e:
        logger.error(f"Validation failed: Input file {cleaned_output_path} not found")
        raise FileNotFoundError(f"Input file {cleaned_output_path} not found. Ensure DataCleaner ran successfully.")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise ValueError(f"Data validation failed: {str(e)}")
    
    # Step 3: Monitor Data Metrics
    logger.info("Running data metrics monitoring")
    try:
        monitor = DataMetricsMonitor(input_path=validated_output_path)
        metrics = monitor.monitor(baseline_path=os.path.join(metrics_dir, 'baseline_metrics.csv'))
        logger.info(f"Data metrics monitoring completed. Metrics saved to {metrics_dir}")
    except FileNotFoundError as e:
        logger.error(f"Metrics monitoring failed: Input file {validated_output_path} not found")
        raise FileNotFoundError(f"Input file {validated_output_path} not found. Ensure DataValidator ran successfully.")
    except Exception as e:
        logger.error(f"Metrics monitoring failed: {str(e)}")
        raise ValueError(f"Data metrics monitoring failed: {str(e)}")
    
    # Step 4: Version Data
    logger.info("Running data versioning")
    try:
        versioning = DataVersioning(
            input_path=cleaned_output_path,
            processed_path=validated_output_path,
            versions_dir=versions_dir,
            lineage_dir=lineage_dir
        )
        processing_steps = [
            'clean_data',
            'validate_data',
            'monitor_metrics'
        ]
        parameters = {
            'input_path': input_path,
            'cleaned_output_path': cleaned_output_path,
            'validated_output_path': validated_output_path,
            'metrics_dir': metrics_dir,
            'versions_dir': versions_dir,
            'lineage_dir': lineage_dir,
            'cleaning_params': {
                'outlier_iqr_threshold': 1.5,
                'winsorize_limits': 0.05,
                'min_text_length': 10
            },
            'validation_params': {
                'price_change_percentage_threshold': 100,
                'daily_alpha_threshold': 1000,
                'outlier_iqr_multiplier': 3.0,
                'class_balance_threshold': 0.2,
                'min_text_length': 10
            },
            'metrics_params': {
                'outlier_iqr_multiplier': 1.5
            }
        }
        version, versioned_path, lineage_path = versioning.version_and_track(processing_steps, parameters)
        logger.info(f"Data versioning completed. Version {version} saved to {versioned_path}")
    except FileNotFoundError as e:
        logger.error(f"Versioning failed: {str(e)}")
        raise FileNotFoundError(f"Versioning failed: {str(e)}. Ensure input and processed files exist.")
    except Exception as e:
        logger.error(f"Versioning failed: {str(e)}")
        raise ValueError(f"Data versioning failed: {str(e)}")
    
    logger.info("Data quality pipeline completed")
    return {
        'cleaned_data_path': cleaned_output_path,
        'validated_data_path': validated_output_path,
        'version': version,
        'versioned_path': versioned_path,
        'lineage_path': lineage_path,
        'metrics': metrics
    }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    result = run_data_quality_pipeline(
        input_path=os.path.join(data_dir, 'backfilling', 'processed_results.csv'),
        cleaned_output_path=os.path.join(data_dir, 'clean', f'cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'),
        validated_output_path=os.path.join(data_dir, 'clean', f'validated_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'),
        metrics_dir=os.path.join(data_dir, 'quality_metrics'),
        versions_dir=os.path.join(data_dir, 'versions'),
        lineage_dir=os.path.join(data_dir, 'lineage')
    )
    logger.info(f"Pipeline completed. Cleaned data saved to {result['cleaned_data_path']}, validated data saved to {result['validated_data_path']}")
