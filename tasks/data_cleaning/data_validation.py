import pandas as pd
import os
import logging
from typing import Dict, Optional
import numpy as np
from datetime import datetime

def setup_logger():
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'data_cleaning', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'validation.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

class DataValidator:
    """Class to validate financial news impact prediction data for Step 1."""
    
    def __init__(
        self,
        input_path: str = f'data/clean/cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv',
        output_path: str = f'data/clean/validated_price_moves_{datetime.now().strftime("%Y%m%d")}.csv',
        metrics_path: str = 'data/quality_metrics/validation_metrics.csv'
    ):
        """
        Initialize DataValidator.

        Args:
            input_path (str): Path to cleaned data from DataCleaner.
            output_path (str): Path to save validated data.
            metrics_path (str): Path to save validation metrics.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.metrics_path = metrics_path
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def check_input_structure(self) -> bool:
        """Validate input data structure."""
        if not os.path.exists(self.input_path):
            logger.error(f"Input file {self.input_path} does not exist. Ensure DataCleaner has run.")
            raise FileNotFoundError(f"{self.input_path} not found. Check if 'cleaned_price_moves_{datetime.now().strftime('%Y%m%d')}.csv' exists in data/clean/.")
        
        try:
            self.df = pd.read_csv(self.input_path, parse_dates=['published_date'], low_memory=False)
            logger.info(f"Loaded {len(self.df)} records from {self.input_path}")
            self.metrics['initial_rows'] = len(self.df)
        except Exception as e:
            logger.error(f"Failed to load {self.input_path}: {str(e)}")
            raise ValueError(f"Failed to load {self.input_path}: {str(e)}")

        expected_columns = [
            'news_id', 'ticker_url', 'title', 'link', 'published_date', 'company', 'event',
            'reason', 'publisher', 'industry', 'publisher_topic', 'instrument_id', 'yf_ticker',
            'published_date_gmt', 'timezone', 'publisher_summary', 'predicted_side',
            'predicted_move', 'content', 'language', 'begin_price', 'end_price',
            'index_begin_price', 'index_end_price', 'price_change', 'price_change_percentage',
            'index_price_change', 'index_price_change_percentage', 'daily_alpha',
            'actual_side', 'volume', 'market'
        ]
        missing_cols = [col for col in expected_columns if col not in self.df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            self.metrics['missing_columns'] = len(missing_cols)
        else:
            logger.info("All expected columns present")
            self.metrics['missing_columns'] = 0
        return len(missing_cols) == 0

    def validate_data_types(self) -> None:
        """Validate and convert data types for expected columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        expected_types = {
            'news_id': str,
            'ticker_url': str,
            'title': str,
            'link': str,
            'published_date': 'datetime64[ns, UTC]',
            'company': str,
            'event': str,
            'reason': str,
            'publisher': str,
            'industry': str,
            'publisher_topic': str,
            'instrument_id': str,
            'yf_ticker': str,
            'published_date_gmt': str,
            'timezone': str,
            'publisher_summary': str,
            'predicted_side': str,
            'predicted_move': str,
            'content': str,
            'language': str,
            'begin_price': float,
            'end_price': float,
            'index_begin_price': float,
            'index_end_price': float,
            'price_change': float,
            'price_change_percentage': float,
            'index_price_change': float,
            'index_price_change_percentage': float,
            'daily_alpha': float,
            'actual_side': str,
            'volume': float,
            'market': str
        }
        
        type_issues = []
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                try:
                    if expected_type == 'datetime64[ns, UTC]':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce', utc=True, format='mixed')
                        invalid_dates = self.df[col].isna()
                        if invalid_dates.sum() > 0:
                            logger.warning(f"Found {invalid_dates.sum()} invalid dates in {col}")
                            self.df.loc[invalid_dates, col] = pd.Timestamp('2000-01-01', tz='UTC')
                            self.metrics[f'invalid_dates_{col}'] = invalid_dates.sum()
                    else:
                        self.df[col] = self.df[col].astype(expected_type, errors='ignore')
                        if col in [
                            'news_id', 'ticker_url', 'title', 'content', 'link', 'company', 'event',
                            'reason', 'publisher', 'industry', 'publisher_topic', 'instrument_id',
                            'yf_ticker', 'published_date_gmt', 'timezone', 'publisher_summary',
                            'predicted_side', 'predicted_move', 'language', 'actual_side', 'market'
                        ]:
                            self.df[col] = self.df[col].fillna('missing')
                        elif col in [
                            'begin_price', 'end_price', 'index_begin_price', 'index_end_price',
                            'price_change', 'price_change_percentage', 'index_price_change',
                            'index_price_change_percentage', 'daily_alpha', 'volume'
                        ]:
                            invalid = self.df[col].isna()
                            if invalid.sum() > 0:
                                median = self.df[col].median()
                                self.df.loc[invalid, col] = median
                                logger.info(f"Imputed {invalid.sum()} missing {col} with median {median}")
                                self.metrics[f'imputed_{col}'] = invalid.sum()
                    actual_type = str(self.df[col].dtype)
                    self.metrics[f'type_valid_{col}'] = actual_type.startswith(str(expected_type).split('[')[0])
                    if not self.metrics[f'type_valid_{col}']:
                        type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
                except Exception as e:
                    type_issues.append(f"{col}: failed conversion to {expected_type}, error: {str(e)}")
                    self.metrics[f'type_valid_{col}'] = False
            else:
                self.metrics[f'type_valid_{col}'] = False
                type_issues.append(f"{col}: missing from dataset")
        
        self.metrics['type_issues'] = len(type_issues)
        if type_issues:
            logger.warning(f"Data type issues: {type_issues}")
        else:
            logger.info("All data types validated successfully")

    def validate_categorical_values(self) -> None:
        """Balance actual_side classes and preserve event distribution."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'actual_side' in self.df.columns:
            class_counts = self.df['actual_side'].value_counts()
            logger.info(f"Initial class distribution:\n{class_counts}")
            
            if 'UP' in class_counts and 'DOWN' in class_counts:
                if abs(class_counts['UP'] - class_counts['DOWN']) / len(self.df) > 0.2:
                    from sklearn.utils import resample
                    df_majority = self.df[self.df['actual_side'] == class_counts.idxmax()]
                    df_minority = self.df[self.df['actual_side'] == class_counts.idxmin()]
                    df_minority_upsampled = resample(
                        df_minority,
                        replace=True,
                        n_samples=len(df_majority),
                        random_state=42
                    )
                    self.df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)
                    logger.info(f"Balanced classes. New distribution:\n{self.df['actual_side'].value_counts()}")
                    self.metrics['class_balance_applied'] = True
                else:
                    logger.info("Class imbalance within acceptable threshold")
                    self.metrics['class_balance_applied'] = False
            else:
                logger.warning("Insufficient classes in actual_side for balancing")
                self.metrics['class_balance_applied'] = False

    def validate_outliers(self) -> None:
        """Use relaxed outlier detection with higher IQR multiplier."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        for col in ['price_change_percentage', 'daily_alpha']:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                outliers = self.df[
                    (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                ]
                self.metrics[f'outliers_{col}'] = len(outliers)
                
                if len(outliers) > 0:
                    logger.warning(f"Found {len(outliers)} outliers in {col} (relaxed bounds)")
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                    logger.info(f"Capped outliers in {col}")
                else:
                    logger.info(f"No outliers detected in {col}")

    def validate_price_ranges(self) -> None:
        """Validate price change percentage and daily alpha ranges."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'price_change_percentage' in self.df.columns:
            invalid_prices = self.df[
                (self.df['price_change_percentage'].abs() > 100) | (self.df['price_change_percentage'].isna())
            ]
            self.metrics['invalid_prices'] = len(invalid_prices)
            if len(invalid_prices) > 0:
                logger.warning(f"Found {len(invalid_prices)} invalid price change percentages")
                median_price = self.df['price_change_percentage'].median()
                self.df.loc[invalid_prices.index, 'price_change_percentage'] = median_price
                logger.info(f"Imputed {len(invalid_prices)} invalid prices with median {median_price}")
            else:
                logger.info("All price change percentages are within valid range")

        if 'daily_alpha' in self.df.columns:
            invalid_alphas = self.df[
                (self.df['daily_alpha'].isna()) | (self.df['daily_alpha'].abs() > 1000)
            ]
            self.metrics['invalid_daily_alpha'] = len(invalid_alphas)
            if len(invalid_alphas) > 0:
                logger.warning(f"Found {len(invalid_alphas)} invalid daily alpha values")
                median_alpha = self.df['daily_alpha'].median()
                self.df.loc[invalid_alphas.index, 'daily_alpha'] = median_alpha
                logger.info(f"Imputed {len(invalid_alphas)} invalid daily alphas with median {median_alpha}")
            else:
                logger.info("All daily alpha values are valid")

    def validate_text_quality(self) -> None:
        """Validate text quality for content and title columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        text_cols = ['title', 'content']
        for col in text_cols:
            if col in self.df.columns:
                empty_texts = self.df[self.df[col].isna() | (self.df[col].str.strip() == '')]
                short_texts = self.df[(self.df[col].str.len() < 10) & (self.df[col].notna())]
                self.metrics[f'empty_{col}'] = len(empty_texts)
                self.metrics[f'short_{col}'] = len(short_texts)
                if len(empty_texts) > 0:
                    logger.warning(f"Found {len(empty_texts)} empty {col} values")
                    self.df.loc[empty_texts.index, col] = 'missing'
                if len(short_texts) > 0:
                    logger.warning(f"Found {len(short_texts)} short {col} values (<10 chars)")
                    self.df.loc[short_texts.index, col] = 'missing'
                if len(empty_texts) == 0 and len(short_texts) == 0:
                    logger.info(f"All {col} values are valid")

    def save_cleaned_data(self) -> None:
        """Save the validated dataset to CSV."""
        if self.df is None:
            raise ValueError("Data not loaded")
        if self.df.empty:
            logger.error("No valid data remains after validation")
            raise ValueError("Output DataFrame is empty")
        try:
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"Saved validated data to {self.output_path} with {len(self.df)} rows")
            self.metrics['final_rows'] = len(self.df)
        except Exception as e:
            logger.error(f"Failed to save validated data to {self.output_path}: {str(e)}")
            raise

    def save_metrics(self) -> None:
        """Save validation metrics to CSV."""
        pd.DataFrame([self.metrics]).to_csv(self.metrics_path, index=False)
        logger.info(f"Saved validation metrics to {self.metrics_path}")

    def validate(self) -> pd.DataFrame:
        """Run the full validation pipeline."""
        logger.info("Starting data validation for Step 1")
        if not self.check_input_structure():
            logger.warning("Input structure issues detected, proceeding with available columns")
        self.validate_data_types()
        self.validate_categorical_values()
        self.validate_price_ranges()
        self.validate_outliers()
        self.validate_text_quality()
        self.save_cleaned_data()
        self.save_metrics()
        logger.info("Data validation pipeline completed")
        return self.df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    validator = DataValidator(
        input_path=os.path.join(data_dir, 'clean', f'cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'),
        output_path=os.path.join(data_dir, 'clean', f'validated_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'),
        metrics_path=os.path.join(data_dir, 'quality_metrics', 'validation_metrics.csv')
    )
    validated_df = validator.validate()
'''
import pandas as pd
import os
import logging
from typing import Dict, Optional
import numpy as np

def setup_logger():
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'data', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'validation.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

class DataValidator:
    """Class to validate financial news impact prediction data for Step 1."""
    
    def __init__(
        self,
        input_path: str,
        output_path: str = 'data/clean/clean_price_moves.csv',
        metrics_path: str = 'data/quality_metrics/validation_metrics.csv'
    ):
        """
        Initialize DataValidator.

        Args:
            input_path (str): Path to input data.
            output_path (str): Path to save cleaned data.
            metrics_path (str): Path to save validation metrics.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.metrics_path = metrics_path
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def check_input_structure(self) -> bool:
        """Validate input data structure."""
        if not os.path.exists(self.input_path):
            logger.error(f"Input file {self.input_path} does not exist")
            raise FileNotFoundError(f"{self.input_path} not found")
        
        try:
            self.df = pd.read_csv(self.input_path, low_memory=False)
            logger.info(f"Loaded {len(self.df)} records from {self.input_path}")
            self.metrics['initial_rows'] = len(self.df)
        except Exception as e:
            logger.error(f"Failed to load {self.input_path}: {str(e)}")
            raise ValueError(f"Failed to load {self.input_path}: {str(e)}")

        expected_columns = [
            'id', 'ticker', 'published_date', 'event', 'actual_side',
            'price_change_percentage', 'daily_alpha', 'content', 'title'
        ]
        missing_cols = [col for col in expected_columns if col not in self.df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            self.metrics['missing_columns'] = len(missing_cols)
        else:
            logger.info("All expected columns present")
            self.metrics['missing_columns'] = 0
        return len(missing_cols) == 0

    def validate_data_types(self) -> None:
        """Validate and convert data types for expected columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        expected_types = {
            'id': str,
            'ticker': str,
            'published_date': 'datetime64[ns, UTC]',
            'event': str,
            'actual_side': str,
            'price_change_percentage': float,
            'daily_alpha': float,
            'content': str,
            'title': str
        }
        
        type_issues = []
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                try:
                    if expected_type == 'datetime64[ns, UTC]':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce', utc=True, format='mixed')
                        invalid_dates = self.df[col].isna()
                        if invalid_dates.sum() > 0:
                            logger.warning(f"Found {invalid_dates.sum()} invalid dates in {col}")
                            self.df.loc[invalid_dates, col] = pd.Timestamp('2000-01-01', tz='UTC')
                            self.metrics[f'invalid_dates_{col}'] = invalid_dates.sum()
                    else:
                        self.df[col] = self.df[col].astype(expected_type, errors='ignore')
                        if col in ['content', 'title']:
                            self.df[col] = self.df[col].fillna('missing')
                        elif col in ['price_change_percentage', 'daily_alpha']:
                            invalid = self.df[col].isna()
                            if invalid.sum() > 0:
                                median = self.df[col].median()
                                self.df.loc[invalid, col] = median
                                logger.info(f"Imputed {invalid.sum()} missing {col} with median {median}")
                                self.metrics[f'imputed_{col}'] = invalid.sum()
                    actual_type = str(self.df[col].dtype)
                    self.metrics[f'type_valid_{col}'] = actual_type.startswith(str(expected_type).split('[')[0])
                    if not self.metrics[f'type_valid_{col}']:
                        type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
                except Exception as e:
                    type_issues.append(f"{col}: failed conversion to {expected_type}, error: {str(e)}")
                    self.metrics[f'type_valid_{col}'] = False
            else:
                self.metrics[f'type_valid_{col}'] = False
                type_issues.append(f"{col}: missing from dataset")
        
        self.metrics['type_issues'] = len(type_issues)
        if type_issues:
            logger.warning(f"Data type issues: {type_issues}")
        else:
            logger.info("All data types validated successfully")

    def validate_categorical_values(self) -> None:
        """Balance actual_side classes and preserve event distribution."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'actual_side' in self.df.columns:
            # Get class counts
            class_counts = self.df['actual_side'].value_counts()
            logger.info(f"Initial class distribution:\n{class_counts}")
            
            # Balance classes if imbalance exceeds threshold
            if abs(class_counts['UP'] - class_counts['DOWN']) / len(self.df) > 0.2:
                from sklearn.utils import resample
                
                # Separate classes
                df_majority = self.df[self.df['actual_side'] == class_counts.idxmax()]
                df_minority = self.df[self.df['actual_side'] == class_counts.idxmin()]
                
                # Upsample minority class
                df_minority_upsampled = resample(
                    df_minority,
                    replace=True,
                    n_samples=len(df_majority),
                    random_state=42
                )
                
                # Combine and shuffle
                self.df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1)
                logger.info(f"Balanced classes. New distribution:\n{self.df['actual_side'].value_counts()}")
                self.metrics['class_balance_applied'] = True
            else:
                logger.info("Class imbalance within acceptable threshold")
                self.metrics['class_balance_applied'] = False

    def validate_outliers(self) -> None:
        """Use relaxed outlier detection with higher IQR multiplier."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'price_change_percentage' in self.df.columns:
            Q1 = self.df['price_change_percentage'].quantile(0.25)
            Q3 = self.df['price_change_percentage'].quantile(0.75)
            IQR = Q3 - Q1
            # Increased from 1.5 to 3.0 for more relaxed bounds
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            outliers = self.df[
                (self.df['price_change_percentage'] < lower_bound) | 
                (self.df['price_change_percentage'] > upper_bound)
            ]
            self.metrics['outliers_price'] = len(outliers)
            
            if len(outliers) > 0:
                logger.warning(f"Found {len(outliers)} price change outliers (relaxed bounds)")
                # Instead of imputing, cap values to preserve variance
                self.df.loc[self.df['price_change_percentage'] < lower_bound, 'price_change_percentage'] = lower_bound
                self.df.loc[self.df['price_change_percentage'] > upper_bound, 'price_change_percentage'] = upper_bound
                logger.info("Capped outliers instead of imputing")

    def validate_price_ranges(self) -> None:
        """Validate price change percentage and daily alpha ranges."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'price_change_percentage' in self.df.columns:
            invalid_prices = self.df[
                (self.df['price_change_percentage'].abs() > 100) |
                (self.df['price_change_percentage'].isna())
            ]
            self.metrics['invalid_prices'] = len(invalid_prices)
            if len(invalid_prices) > 0:
                logger.warning(f"Found {len(invalid_prices)} invalid price change percentages")
                median_price = self.df['price_change_percentage'].median()
                self.df.loc[invalid_prices.index, 'price_change_percentage'] = median_price
                logger.info(f"Imputed {len(invalid_prices)} invalid prices with median {median_price}")
            else:
                logger.info("All price change percentages are within valid range")

        if 'daily_alpha' in self.df.columns:
            invalid_alphas = self.df[
                (self.df['daily_alpha'].isna()) |
                (self.df['daily_alpha'].abs() > 1000)
            ]
            self.metrics['invalid_daily_alpha'] = len(invalid_alphas)
            if len(invalid_alphas) > 0:
                logger.warning(f"Found {len(invalid_alphas)} invalid daily alpha values")
                median_alpha = self.df['daily_alpha'].median()
                self.df.loc[invalid_alphas.index, 'daily_alpha'] = median_alpha
                logger.info(f"Imputed {len(invalid_alphas)} invalid daily alphas with median {median_alpha}")
            else:
                logger.info("All daily alpha values are valid")


    def validate_text_quality(self) -> None:
        """Validate text quality for content and title columns."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        text_cols = ['content', 'title']
        for col in text_cols:
            if col in self.df.columns:
                empty_texts = self.df[self.df[col].isna() | (self.df[col].str.strip() == '')]
                short_texts = self.df[(self.df[col].str.len() < 10) & (self.df[col].notna())]
                self.metrics[f'empty_{col}'] = len(empty_texts)
                self.metrics[f'short_{col}'] = len(short_texts)
                if len(empty_texts) > 0:
                    logger.warning(f"Found {len(empty_texts)} empty {col} values")
                    self.df.loc[empty_texts.index, col] = 'missing'
                if len(short_texts) > 0:
                    logger.warning(f"Found {len(short_texts)} short {col} values (<10 chars)")
                    self.df.loc[short_texts.index, col] = 'missing'
                if len(empty_texts) == 0 and len(short_texts) == 0:
                    logger.info(f"All {col} values are valid")

    def save_cleaned_data(self) -> None:
        """Save the cleaned dataset to CSV."""
        if self.df is None:
            raise ValueError("Data not loaded")
        if self.df.empty:
            logger.error("No valid data remains after validation")
            raise ValueError("Output DataFrame is empty")
        try:
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"Saved cleaned data to {self.output_path} with {len(self.df)} rows")
            self.metrics['final_rows'] = len(self.df)
        except Exception as e:
            logger.error(f"Failed to save cleaned data to {self.output_path}: {str(e)}")
            raise

    def save_metrics(self) -> None:
        """Save validation metrics to CSV."""
        pd.DataFrame([self.metrics]).to_csv(self.metrics_path, index=False)
        logger.info(f"Saved validation metrics to {self.metrics_path}")

    def validate(self) -> pd.DataFrame:
        """Run the full validation pipeline."""
        logger.info("Starting data validation for Step 1")
        if not self.check_input_structure():
            logger.warning("Input structure issues detected, proceeding with available columns")
        self.validate_data_types()
        self.validate_categorical_values()
        self.validate_price_ranges()
        self.validate_outliers()
        self.validate_text_quality()
        self.save_cleaned_data()
        self.save_metrics()
        logger.info("Data validation pipeline completed")
        return self.df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    validator = DataValidator(
        input_path=os.path.join(data_dir, 'all_price_moves.csv'),
        output_path=os.path.join(data_dir, 'clean', 'clean_price_moves.csv'),
        metrics_path=os.path.join(data_dir, 'quality_metrics', 'validation_metrics.csv')
    )
    validated_df = validator.validate()
    '''