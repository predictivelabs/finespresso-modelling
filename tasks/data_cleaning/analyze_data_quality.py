import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class DataQualityAnalyzer:
    def __init__(self, filepath='data/backfilling/processed_results.csv', output_dir='data/quality_reports'):
        self.filepath = filepath
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def link_news_to_price_moves(self):
        """Analyze linkage between news articles and price moves using instrument_id and yf_ticker."""
        print("\n=== Linking News to Price Moves ===")
        news_count = len(self.df)
        price_move_count = self.df[['begin_price', 'end_price']].notna().all(axis=1).sum()
        print(f"Total news articles: {news_count}")
        print(f"Total price moves: {price_move_count}")

        # Identify missing price moves
        missing_price_moves = self.df[self.df['begin_price'].isna() | self.df['end_price'].isna()]
        unique_tickers = missing_price_moves['yf_ticker'].dropna().unique()
        print(f"Unique tickers with missing price moves: {len(unique_tickers)}")

    def validate_price_movements(self):
        """Validate price movement calculations with dynamic column handling."""
        required_cols = ['begin_price', 'end_price', 'price_change', 'price_change_percentage']
        if all(col in self.df.columns for col in required_cols):
            tolerance = 1e-6
            self.df['calc_price_change'] = self.df['end_price'] - self.df['begin_price']
            self.df['calc_price_change_percentage'] = (self.df['end_price'] - self.df['begin_price']) / self.df['begin_price'] * 100
            validation_report = pd.DataFrame({
                'Metric': ['Price Change', 'Price Change %'],
                'Correct (%)': [
                    (np.abs(self.df['price_change'] - self.df['calc_price_change']) < tolerance).mean() * 100,
                    (np.abs(self.df['price_change_percentage'] - self.df['calc_price_change_percentage']) < tolerance).mean() * 100
                ]
            })
            index_cols = ['index_begin_price', 'index_end_price', 'index_price_change', 'index_price_change_percentage', 'daily_alpha']
            if all(col in self.df.columns for col in index_cols):
                self.df['calc_index_price_change'] = self.df['index_end_price'] - self.df['index_begin_price']
                self.df['calc_index_price_change_percentage'] = (self.df['index_end_price'] - self.df['index_begin_price']) / self.df['index_begin_price'] * 100
                self.df['calc_daily_alpha'] = self.df['calc_price_change_percentage'] - self.df['calc_index_price_change_percentage']
                validation_report = pd.concat([validation_report, pd.DataFrame({
                    'Metric': ['Index Price Change', 'Index Price Change %', 'Daily Alpha'],
                    'Correct (%)': [
                        (np.abs(self.df['index_price_change'] - self.df['calc_index_price_change']) < tolerance).mean() * 100,
                        (np.abs(self.df['index_price_change_percentage'] - self.df['calc_index_price_change_percentage']) < tolerance).mean() * 100,
                        (np.abs(self.df['daily_alpha'] - self.df['calc_daily_alpha']) < tolerance).mean() * 100
                    ]
                })], ignore_index=True)
            validation_report.to_csv(os.path.join(self.output_dir, 'price_validation_report.csv'), index=False)
            print("\n=== Price Movement Validation Report ===")
            print(validation_report)

    def compute_quality_metrics(self):
        """Compute data quality metrics."""
        metrics = {
            'completeness': (1 - self.df.isnull().mean().mean()) * 100,
            'duplicate_rate': self.df.duplicated().mean() * 100,
        }
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metrics_df.to_csv(os.path.join(self.output_dir, 'quality_metrics.csv'), index=False)
        return metrics

    def analyze(self):
        """Comprehensive data quality analysis with CSV output."""
        # Load data
        self.df = pd.read_csv(self.filepath)
        print("=== Initial Data Overview ===")
        print(f"Total records: {len(self.df)}")
        print(f"Columns: {self.df.columns.tolist()}")

        # 1. Link News to Price Moves
        self.link_news_to_price_moves()

        # 2. Missing Values Analysis
        print("\n=== Missing Values Analysis ===")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_report = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        print(missing_report.sort_values('Missing Percentage', ascending=False))
        missing_report.to_csv(os.path.join(self.output_dir, 'missing_values_report.csv'), index=False)

        # 3. Duplicate Analysis
        print("\n=== Duplicate Records Analysis ===")
        duplicates = self.df.duplicated().sum()
        print(f"Exact duplicates: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        duplicate_report = pd.DataFrame({
            'Metric': ['Total Records', 'Duplicates', 'Duplicate Percentage'],
            'Value': [len(self.df), duplicates, duplicates/len(self.df)*100]
        })
        duplicate_report.to_csv(os.path.join(self.output_dir, 'duplicate_report.csv'), index=False)

        # 4. Price Movement Analysis
        print("\n=== Price Movement Analysis ===")
        if 'price_change_percentage' in self.df.columns:
            print(self.df['price_change_percentage'].describe())
            plt.figure(figsize=(12, 6))
            sns.histplot(self.df['price_change_percentage'], bins=50)
            plt.title('Distribution of Price Change Percentage')
            plt.xlabel('Price Change Percentage')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(self.output_dir, 'price_change_distribution.png'))
            plt.close()
        else:
            print("Price change percentage column not found.")

        # 5. Event Type Analysis
        print("\n=== Event Type Distribution ===")
        if 'event' in self.df.columns:
            event_counts = self.df['event'].value_counts(dropna=False)
            print(event_counts)
            event_counts.to_csv(os.path.join(self.output_dir, 'event_distribution.csv'))
        else:
            print("Event column not found.")

        # 6. Date Analysis
        print("\n=== Published Date Analysis ===")
        if 'published_date' in self.df.columns:
            self.df['published_date'] = pd.to_datetime(self.df['published_date'], errors='coerce')
            print(f"Date range: {self.df['published_date'].min()} to {self.df['published_date'].max()}")
            print(f"Missing dates: {self.df['published_date'].isnull().sum()}")
            date_report = pd.DataFrame({
                'Column': ['published_date'],
                'Min Date': [self.df['published_date'].min()],
                'Max Date': [self.df['published_date'].max()],
                'Missing Dates': [self.df['published_date'].isnull().sum()]
            })
            date_report.to_csv(os.path.join(self.output_dir, 'date_analysis_report.csv'), index=False)
        else:
            print("Published date column not found.")

        # 7. Price Movement Validation
        self.validate_price_movements()

        # 8. Compute Quality Metrics
        print("\n=== Data Quality Metrics ===")
        quality_metrics = self.compute_quality_metrics()
        print(quality_metrics)

        return self.df

# Usage example
if __name__ == '__main__':
    analyzer = DataQualityAnalyzer()
    df = analyzer.analyze()
