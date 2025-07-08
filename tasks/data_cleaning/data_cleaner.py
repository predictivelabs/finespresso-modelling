import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
import re
from datetime import datetime
import spacy

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'data_cleaning', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'cleaning.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

class DataCleaner:
    """Class to clean financial news impact prediction data with event-specific handling."""
    
    def __init__(
        self,
        input_path: str = 'data/backfillin/processed_results.csv',
        output_path: str = 'data/clean/cleaned_price_moves.csv',
        metrics_path: str = 'data/quality_metrics/cleaning_metrics.csv'
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.metrics_path = metrics_path
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Dict = {}
        self.event_thresholds = {
            'changes_in_companys_own_shares': 1.5,
            'financial_results': 1.5,
            'bond_fixing': 1.5,
            'annual_general_meeting': 1.5,
            'earnings_releases_and_operating_results': 1.5,
            'default': 1.5
        }
        self.nlp_models = {}
        self.load_spacy_models()

    def load_spacy_models(self):
        """Load spaCy models for supported languages with error handling."""
        supported_languages = {
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "no": "nb_core_news_sm"
        }
        for lang, model_name in supported_languages.items():
            try:
                self.nlp_models[lang] = spacy.load(model_name, disable=["parser", "ner"])
                logger.info(f"Loaded spaCy model for {lang}: {model_name}")
            except OSError:
                logger.warning(f"Model {model_name} not found for language {lang}. Install it with 'python -m spacy download {model_name}'")
                self.nlp_models[lang] = None
        self.default_nlp = self.nlp_models.get("en")

    def clean_text(self, text: str, lang: str = "en") -> str:
        """Clean and preprocess text using spaCy."""
        if pd.isna(text):
            return "missing"
        nlp = self.nlp_models.get(lang, self.default_nlp)
        if nlp is None:
            logger.warning(f"No model for language {lang}, using English model as fallback")
            nlp = self.default_nlp
        doc = nlp(str(text))
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def validate_price_movements(self):
        """Validate price movement calculations and store metrics."""
        required_cols = ['begin_price', 'end_price', 'price_change', 'price_change_percentage']
        if all(col in self.df.columns for col in required_cols):
            tolerance = 1e-6
            self.df["calc_price_change"] = self.df["end_price"] - self.df["begin_price"]
            self.df["calc_price_change_percentage"] = (self.df["end_price"] - self.df["begin_price"]) / self.df["begin_price"] * 100
            validation_report = pd.DataFrame({
                'Metric': ['Price Change', 'Price Change %'],
                'Correct (%)': [
                    (np.abs(self.df["price_change"] - self.df["calc_price_change"]) < tolerance).mean() * 100,
                    (np.abs(self.df["price_change_percentage"] - self.df["calc_price_change_percentage"]) < tolerance).mean() * 100
                ]
            })
            index_cols = ['index_begin_price', 'index_end_price', 'index_price_change', 'index_price_change_percentage', 'daily_alpha']
            if all(col in self.df.columns for col in index_cols):
                self.df["calc_index_price_change"] = self.df["index_end_price"] - self.df["index_begin_price"]
                self.df["calc_index_price_change_percentage"] = (self.df["index_end_price"] - self.df["index_begin_price"]) / self.df["index_begin_price"] * 100
                self.df["calc_daily_alpha"] = self.df["calc_price_change_percentage"] - self.df["calc_index_price_change_percentage"]
                validation_report = pd.concat([validation_report, pd.DataFrame({
                    'Metric': ['Index Price Change', 'Index Price Change %', 'Daily Alpha'],
                    'Correct (%)': [
                        (np.abs(self.df["index_price_change"] - self.df["calc_index_price_change"]) < tolerance).mean() * 100,
                        (np.abs(self.df["index_price_change_percentage"] - self.df["calc_index_price_change_percentage"]) < tolerance).mean() * 100,
                        (np.abs(self.df["daily_alpha"] - self.df["calc_daily_alpha"]) < tolerance).mean() * 100
                    ]
                })], ignore_index=True)
            logger.info(f"Price movement validation results:\n{validation_report}")
            validation_report.to_csv(os.path.join(os.path.dirname(self.metrics_path), 'price_validation_report.csv'), index=False)
            self.metrics['price_validation_accuracy'] = validation_report['Correct (%)'].mean()
            # Drop calculated columns
            calc_cols = [
                'calc_price_change', 'calc_price_change_percentage',
                'calc_index_price_change', 'calc_index_price_change_percentage', 'calc_daily_alpha'
            ]
            self.df = self.df.drop(columns=[col for col in calc_cols if col in self.df.columns], errors='ignore')

    def load_data(self) -> None:
        """Load data from CSV file with error handling."""
        try:
            self.df = pd.read_csv(self.input_path, parse_dates=['published_date'], low_memory=False)
            logger.info(f"Loaded {len(self.df)} records from {self.input_path}")
            self.metrics['initial_rows'] = len(self.df)
            self.metrics['initial_columns'] = len(self.df.columns)
        except Exception as e:
            logger.error(f"Failed to load {self.input_path}: {str(e)}")
            raise ValueError(f"Failed to load {self.input_path}: {str(e)}")

    def remove_rows_missing_critical(self) -> None:
        """Remove rows where both actual_side and price_change_percentage are missing."""
        if self.df is None:
            raise ValueError("Data not loaded")
        initial_rows = len(self.df)
        if 'actual_side' in self.df.columns and 'price_change_percentage' in self.df.columns:
            self.df = self.df.dropna(subset=['actual_side', 'price_change_percentage'], how='all')
            removed_rows = initial_rows - len(self.df)
            logger.info(f"Removed {removed_rows} rows where both actual_side and price_change_percentage are missing")
            self.metrics['removed_rows_missing_critical'] = removed_rows
        else:
            logger.warning("One or both of 'actual_side' and 'price_change_percentage' not in DataFrame")

    def drop_extracted_yf_ticker(self) -> None:
        """Drop the extracted_yf_ticker column."""
        if self.df is None:
            raise ValueError("Data not loaded")
        if 'extracted_yf_ticker' in self.df.columns:
            self.df = self.df.drop(columns=['extracted_yf_ticker'])
            logger.info("Dropped extracted_yf_ticker column")
            self.metrics['dropped_columns'] = ['extracted_yf_ticker']
        else:
            logger.warning("extracted_yf_ticker column not found")

    def clean_event_column(self) -> None:
        """Handle problematic event values (long text or missing)."""
        if self.df is None:
            raise ValueError("Data not loaded")
        if 'event' not in self.df.columns:
            logger.warning("Event column not found")
            return
        
        initial_rows = len(self.df)
        # Identify long text events (e.g., containing "Without specific content")
        long_text_mask = self.df['event'].str.contains("Without specific content", na=False, case=False)
        long_text_count = long_text_mask.sum()
        if long_text_count > 0:
            logger.warning(f"Found {long_text_count} rows with long text in event column")
            # Try to impute with mode of non-long-text events
            valid_events = self.df[~long_text_mask]['event'].dropna()
            mode_event = valid_events.mode()[0] if not valid_events.mode().empty else 'missing'
            if mode_event != 'missing':
                self.df.loc[long_text_mask, 'event'] = mode_event
                logger.info(f"Imputed {long_text_count} long-text events with mode: {mode_event}")
                self.metrics['imputed_long_text_events'] = long_text_count
                self.metrics['imputed_event_value'] = mode_event
            else:
                # If mode is not viable, drop rows
                self.df = self.df[~long_text_mask]
                logger.info(f"Dropped {long_text_count} rows with long-text events")
                self.metrics['dropped_long_text_events'] = long_text_count
        
        # Impute remaining missing events
        missing_event_mask = self.df['event'].isna()
        missing_event_count = missing_event_mask.sum()
        if missing_event_count > 0:
            mode_event = self.df['event'].mode()[0] if not self.df['event'].mode().empty else 'missing'
            self.df.loc[missing_event_mask, 'event'] = mode_event
            logger.info(f"Imputed {missing_event_count} missing events with mode: {mode_event}")
            self.metrics['imputed_missing_events'] = missing_event_count
            self.metrics['imputed_missing_event_value'] = mode_event
        
        removed_rows = initial_rows - len(self.df)
        if removed_rows > 0:
            self.metrics['removed_rows_event_cleaning'] = removed_rows

    def impute_high_missing_columns(self, threshold: float = 0.9) -> None:
        """Impute columns with missing values above threshold using KNN and mode."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        missing_rates = self.df.isna().mean()
        high_missing_cols = missing_rates[missing_rates > threshold].index.tolist()
        
        if high_missing_cols:
            numerical_cols = self.df[high_missing_cols].select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = self.df[high_missing_cols].select_dtypes(include=['object']).columns
            
            if len(numerical_cols) > 0:
                imputer = KNNImputer(n_neighbors=5, weights='uniform')
                self.df[numerical_cols] = pd.DataFrame(
                    imputer.fit_transform(self.df[numerical_cols]),
                    columns=numerical_cols,
                    index=self.df.index
                )
                logger.info(f"KNN imputed numerical columns: {numerical_cols}")
                self.metrics['knn_imputed_columns'] = list(numerical_cols)
            
            for col in categorical_cols:
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'missing'
                self.df[col] = self.df[col].fillna(mode_value)
                logger.info(f"Imputed {col} with mode {mode_value}")
                self.metrics[f'imputed_{col}'] = mode_value

    def impute_missing_values(self) -> None:
        """Impute remaining missing values using event-specific medians for numerical and mode for categorical."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if 'event' in self.df.columns and len(numerical_cols) > 0:
            for col in numerical_cols:
                for event in self.df['event'].unique():
                    mask = self.df['event'] == event
                    if mask.sum() > 0:
                        median_value = self.df.loc[mask, col].median()
                        self.df.loc[mask & self.df[col].isna(), col] = median_value
                        logger.info(f"Imputed {col} for event {event} with median {median_value}")
                        self.metrics[f'imputed_{col}_{event}'] = median_value
        
        for col in categorical_cols:
            if self.df[col].isna().any():
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'missing'
                self.df[col] = self.df[col].fillna(mode_value)
                logger.info(f"Imputed {col} with mode {mode_value}")
                self.metrics[f'imputed_{col}'] = mode_value

    def detect_outliers_iqr(self, columns: List[str]) -> Dict[str, pd.Series]:
        """Detect outliers using event-specific IQR thresholds."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        outlier_flags = {}
        for col in columns:
            if col in self.df.columns:
                for event in self.df['event'].unique():
                    mask = self.df['event'] == event
                    if mask.sum() < 10:
                        continue
                    threshold = self.event_thresholds.get(event, self.event_thresholds['default'])
                    Q1 = self.df.loc[mask, col].quantile(0.25)
                    Q3 = self.df.loc[mask, col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    is_outlier = (self.df.loc[mask, col] < lower_bound) | (self.df.loc[mask, col] > upper_bound)
                    self.df[f'is_outlier_{col}'] = is_outlier.astype(int)
                    outlier_flags[f'{col}_{event}'] = is_outlier
                    logger.info(f"Detected {is_outlier.sum()} outliers in {col} for event {event}")
                    self.metrics[f'outliers_{col}_{event}'] = is_outlier.sum()
        # Drop outlier flag columns
        outlier_cols = [f'is_outlier_{col}' for col in columns if f'is_outlier_{col}' in self.df.columns]
        self.df = self.df.drop(columns=outlier_cols, errors='ignore')
        return outlier_flags

    def handle_outliers(self) -> None:
        """Handle outliers with relaxed winsorizing limits."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        numerical_cols = ['price_change_percentage', 'daily_alpha']
        for col in numerical_cols:
            if col in self.df.columns:
                self.df[col] = winsorize(self.df[col].values, limits=[0.05, 0.05])
                logger.info(f"Relaxed winsorizing (5%) applied to {col}")
                self.metrics[f'winsorized_{col}'] = '5%'

    def clean_datetime(self) -> None:
        """Clean and standardize datetime columns with flexible parsing."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if 'published_date' in self.df.columns:
            try:
                self.df['published_date'] = pd.to_datetime(
                    self.df['published_date'], errors='coerce', utc=True, format='mixed'
                )
                invalid_dates = self.df['published_date'].isna()
                if invalid_dates.any():
                    logger.warning(f"Found {invalid_dates.sum()} invalid published_date values")
                    median_date = self.df['published_date'].median()
                    self.df['published_date'] = self.df['published_date'].fillna(median_date)
                    logger.info(f"Imputed {invalid_dates.sum()} invalid dates with median: {median_date}")
                    self.metrics['invalid_dates'] = invalid_dates.sum()
                    self.metrics['imputed_date'] = str(median_date)
            except Exception as e:
                logger.error(f"Datetime cleaning failed: {str(e)}")
                raise

    def clean_text_data(self) -> None:
        """Clean text data with financial-specific handling and spaCy preprocessing."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        text_cols = ['title', 'content']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: re.sub(r'\$[A-Za-z]+|\d+%|[^\w\s]', ' ', str(x)) if pd.notnull(x) else 'missing'
                )
                self.df[col] = self.df[col].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
                empty_mask = (self.df[col] == '') | (self.df[col].isna())
                if empty_mask.any():
                    self.df.loc[empty_mask, col] = 'missing'
                    logger.info(f"Replaced {empty_mask.sum()} empty {col} values with 'missing'")
                    self.metrics[f'empty_{col}'] = empty_mask.sum()
                
                short_mask = (self.df[col].str.len() < 10) & (self.df[col].notna())
                if short_mask.any():
                    self.df.loc[short_mask, col] = 'missing'
                    logger.info(f"Replaced {short_mask.sum()} short {col} values with 'missing'")
                    self.metrics[f'short_replaced_{col}'] = short_mask.sum()
                
                self.df[col] = self.df.apply(
                    lambda row: self.clean_text(row[col], row['language'] if 'language' in self.df.columns else 'en'), axis=1
                )
                logger.info(f"Applied spaCy preprocessing to {col}")

    def generate_reports(self) -> None:
        """Generate data quality reports."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        # Missing Values Report
        missing_data = self.df.isna().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_report = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        missing_report.to_csv(os.path.join(os.path.dirname(self.metrics_path), 'missing_values_report.csv'), index=False)
        logger.info("Generated missing values report")
        
        # Dataset Shape Report
        shape_report = pd.DataFrame({
            'Metric': ['Rows', 'Columns'],
            'Value': [len(self.df), len(self.df.columns)]
        })
        shape_report.to_csv(os.path.join(os.path.dirname(self.metrics_path), 'shape_report.csv'), index=False)
        logger.info(f"Dataset shape: {len(self.df)} rows, {len(self.df.columns)} columns")

    def save_cleaned_data(self) -> None:
        """Save cleaned data and metrics to CSV."""
        if self.df is None:
            raise ValueError("Data not loaded")
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        logger.info(f"Saved cleaned data to {self.output_path}")
        
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        pd.DataFrame([self.metrics]).to_csv(self.metrics_path, index=False)
        logger.info(f"Saved cleaning metrics to {self.metrics_path}")

    def clean(self) -> pd.DataFrame:
        """Run the full cleaning pipeline."""
        logger.info("Starting data cleaning pipeline")
        self.load_data()
        self.remove_rows_missing_critical()
        self.drop_extracted_yf_ticker()
        self.clean_event_column()
        self.impute_high_missing_columns()
        self.impute_missing_values()
        self.clean_datetime()
        self.validate_price_movements()
        self.detect_outliers_iqr(['price_change_percentage', 'daily_alpha'])
        self.handle_outliers()
        self.clean_text_data()
        self.generate_reports()
        self.save_cleaned_data()
        logger.info("Data cleaning pipeline completed")
        return self.df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    cleaner = DataCleaner(
        input_path=os.path.join(data_dir, 'backfilling', 'processed_results.csv'),
        output_path=os.path.join(data_dir, 'clean', f'cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'),
        metrics_path=os.path.join(data_dir, 'quality_metrics', 'cleaning_metrics.csv')
    )
    cleaned_df = cleaner.clean()

