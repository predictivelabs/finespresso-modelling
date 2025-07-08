import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'data_cleaning', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'metrics.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

class DataMetricsMonitor:
    """Class to monitor data quality metrics for cleaned financial news impact prediction data."""
    
    def __init__(self, input_path: str = f'data/clean/cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'):
        """
        Initialize DataMetricsMonitor.

        Args:
            input_path (str): Path to cleaned data.
        """
        self.input_path = input_path
        try:
            self.df = pd.read_csv(input_path, parse_dates=['published_date'], low_memory=False)
            logger.info(f"Loaded {len(self.df)} records from {input_path}")
        except Exception as e:
            logger.error(f"Failed to load {input_path}: {str(e)}")
            raise ValueError(f"Failed to load {input_path}: {str(e)}")
        self.metrics: Dict = {}
        self.plots_dir = 'data/quality_metrics/plots'
        os.makedirs(self.plots_dir, exist_ok=True)

    def calculate_completeness(self) -> None:
        """Calculate data completeness."""
        completeness = 1 - self.df.isna().mean()
        self.metrics['completeness'] = completeness.to_dict()
        logger.info("Calculated completeness metrics")
        # Save completeness report
        completeness_df = pd.DataFrame({
            'Column': completeness.index,
            'Completeness (%)': completeness.values * 100
        })
        completeness_path = os.path.join(os.path.dirname(self.plots_dir), f'completeness_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        completeness_df.to_csv(completeness_path, index=False)
        logger.info(f"Saved completeness report to {completeness_path}")

    def calculate_consistency(self) -> None:
        """Calculate data consistency for categorical columns."""
        categorical_cols = [
            'news_id', 'ticker_url', 'title', 'content', 'link', 'company', 'event',
            'reason', 'publisher', 'industry', 'publisher_topic', 'instrument_id',
            'yf_ticker', 'published_date_gmt', 'timezone', 'publisher_summary',
            'predicted_side', 'predicted_move', 'language', 'actual_side', 'market'
        ]
        for col in categorical_cols:
            if col in self.df.columns:
                unique_values = self.df[col].nunique()
                self.metrics[f'consistency_unique_{col}'] = unique_values
                logger.info(f"Unique values in {col}: {unique_values}")
        logger.info("Calculated consistency metrics")
        # Save consistency report
        consistency_data = {k: v for k, v in self.metrics.items() if k.startswith('consistency_unique_')}
        consistency_df = pd.DataFrame({
            'Column': [k.replace('consistency_unique_', '') for k in consistency_data.keys()],
            'Unique Values': list(consistency_data.values())
        })
        consistency_path = os.path.join(os.path.dirname(self.plots_dir), f'consistency_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        consistency_df.to_csv(consistency_path, index=False)
        logger.info(f"Saved consistency report to {consistency_path}")

    def calculate_validity(self) -> None:
        """Calculate data validity for numerical columns."""
        numerical_cols = ['begin_price', 'end_price', 'index_begin_price', 'index_end_price', 'volume']
        for col in numerical_cols:
            if col in self.df.columns:
                invalid = (self.df[col] < 0).sum()
                self.metrics[f'validity_invalid_{col}'] = invalid
                logger.info(f"Invalid (negative) values in {col}: {invalid}")
        logger.info("Calculated validity metrics")
        # Save validity report
        validity_data = {k: v for k, v in self.metrics.items() if k.startswith('validity_invalid_')}
        validity_df = pd.DataFrame({
            'Column': [k.replace('validity_invalid_', '') for k in validity_data.keys()],
            'Invalid Values': list(validity_data.values())
        })
        validity_path = os.path.join(os.path.dirname(self.plots_dir), f'validity_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        validity_df.to_csv(validity_path, index=False)
        logger.info(f"Saved validity report to {validity_path}")

    def calculate_outliers(self) -> None:
        """Calculate outliers using IQR for numerical columns."""
        numerical_cols = ['price_change_percentage', 'daily_alpha', 'index_price_change_percentage']
        for col in numerical_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                self.metrics[f'outliers_{col}'] = outliers
                logger.info(f"Outliers in {col}: {outliers}")
        logger.info("Calculated outliers metrics")
        # Save outliers report
        outlier_data = {k: v for k, v in self.metrics.items() if k.startswith('outliers_')}
        outlier_df = pd.DataFrame({
            'Column': [k.replace('outliers_', '') for k in outlier_data.keys()],
            'Outlier Count': list(outlier_data.values())
        })
        outlier_path = os.path.join(os.path.dirname(self.plots_dir), f'outliers_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        outlier_df.to_csv(outlier_path, index=False)
        logger.info(f"Saved outliers report to {outlier_path}")

    def plot_distributions(self) -> None:
        """Plot distributions of numerical columns."""
        numerical_cols = [
            'begin_price', 'end_price', 'index_begin_price', 'index_end_price',
            'price_change', 'price_change_percentage', 'index_price_change',
            'index_price_change_percentage', 'daily_alpha', 'volume'
        ]
        for col in numerical_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plot_path = os.path.join(self.plots_dir, f'distribution_{col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Saved distribution plot for {col} to {plot_path}")

    def compare_with_baseline(self, baseline_path: Optional[str]) -> None:
        """Compare metrics with baseline."""
        if baseline_path and os.path.exists(baseline_path):
            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_metrics = baseline_df.to_dict('records')[0]
                for key in self.metrics:
                    if key in baseline_metrics:
                        if isinstance(self.metrics[key], dict):
                            # Handle nested dicts (e.g., completeness)
                            for sub_key in self.metrics[key]:
                                if sub_key in baseline_metrics.get(key, {}):
                                    diff = abs(self.metrics[key][sub_key] - baseline_metrics[key][sub_key])
                                    self.metrics[f'diff_baseline_{key}_{sub_key}'] = diff
                                    logger.info(f"Difference in {key}_{sub_key} from baseline: {diff}")
                        elif isinstance(self.metrics[key], (int, float)):
                            diff = abs(self.metrics[key] - baseline_metrics[key])
                            self.metrics[f'diff_baseline_{key}'] = diff
                            logger.info(f"Difference in {key} from baseline: {diff}")
                logger.info("Compared metrics with baseline")
            except Exception as e:
                logger.error(f"Failed to compare with baseline {baseline_path}: {str(e)}")
        else:
            logger.warning("Baseline path not provided or does not exist")

    def save_metrics(self, output_path: str) -> None:
        """Save metrics to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Flatten nested dictionaries for CSV
        flat_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f"{key}_{sub_key}"] = sub_value
            else:
                flat_metrics[key] = value
        pd.DataFrame([flat_metrics]).to_csv(output_path, index=False)
        logger.info(f"Saved metrics to {output_path}")

    def monitor(self, baseline_path: Optional[str] = None) -> Dict:
        """Run the full metrics pipeline."""
        logger.info("Starting data metrics monitoring")
        self.calculate_completeness()
        self.calculate_consistency()
        self.calculate_validity()
        self.calculate_outliers()
        self.plot_distributions()
        self.compare_with_baseline(baseline_path)
        output_path = os.path.join(os.path.dirname(self.plots_dir), f'metrics_{datetime.now().strftime("%Y%m%d_%H26H%M%S")}.csv')
        self.save_metrics(output_path)
        logger.info("Data metrics monitoring completed")
        return self.metrics

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    monitor = DataMetricsMonitor(
        input_path=os.path.join(data_dir, 'clean', f'cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv')
    )
    metrics = monitor.monitor(
        baseline_path=os.path.join(data_dir, 'quality_metrics', 'baseline_metrics.csv')
    )
