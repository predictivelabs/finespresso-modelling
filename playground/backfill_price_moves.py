import sys
import pandas as pd
import yfinance as yf
import json
import time as time_module
from datetime import datetime, time
import logging
from typing import Optional, Dict
import openai
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Configure logging
log_dir = os.path.join('playground', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'backfill_data.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized for news_backfill_optimized")

class PriceDataProcessor:
    def __init__(self, openai_api_key: str = None, index_symbol: str = 'SPY', max_workers: int = 10):
        self.index_symbol = index_symbol
        self.max_workers = max_workers
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.processed_count = 0
        self.total_to_process = 0
        self.checkpoint_interval = 1000
        self.intermediate_csv_interval = 1000
        self.checkpoint_file = "data/backfilling/processing_checkpoint.pkl"
        self.output_file = "data/backfilling/processed_results_remaining.csv"

    def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid by trying to get info from yfinance."""
        try:
            base_ticker = ticker.split('.')[0]
            info = yf.Ticker(base_ticker).info
            return info is not None and 'symbol' in info
        except Exception as e:
            logger.warning(f"Failed to validate ticker {ticker}: {e}")
            return False

    def get_previous_trading_day(self, date):
        """Get the previous trading day (Monday-Friday)."""
        date = date - pd.Timedelta(days=1)
        while date.weekday() >= 5:  # Saturday or Sunday
            date = date - pd.Timedelta(days=1)
        return date

    def get_next_trading_day(self, date):
        """Get the next trading day (Monday-Friday)."""
        date = date + pd.Timedelta(days=1)
        while date.weekday() >= 5:  # Saturday or Sunday
            date = date + pd.Timedelta(days=1)
        return date

    def get_price_data(self, ticker: str, published_date: datetime, news_id: str, retries: int = 3) -> Optional[Dict]:
        """Get price data for ticker around the published date with retries."""
        try:
            if not self.validate_ticker(ticker):
                logger.warning(f"Skipping invalid ticker: {ticker}")
                return None

            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=None)

            pub_time = published_date.time()
            pub_date = published_date.date()

            if time(9, 30) <= pub_time < time(16, 0):
                market = 'regular_market'
            elif time(16, 0) <= pub_time:
                market = 'after_market'
            else:
                market = 'pre_market'

            previous_trading_day = self.get_previous_trading_day(pub_date)
            next_trading_day = self.get_next_trading_day(pub_date)

            yf_prev_date = previous_trading_day.strftime('%Y-%m-%d')
            yf_today_date = pub_date.strftime('%Y-%m-%d')
            yf_next_date = next_trading_day.strftime('%Y-%m-%d')

            logger.debug(f"Getting price data for {ticker} on {yf_today_date}, market: {market}")

            for attempt in range(retries):
                try:
                    data = yf.download(ticker, start=yf_prev_date, end=yf_next_date, interval='1d', auto_adjust=False)
                    index_data = yf.download(self.index_symbol, start=yf_prev_date, end=yf_next_date, interval='1d', auto_adjust=False)

                    if data.empty or index_data.empty:
                        logger.warning(f"No price data available for {ticker} on {yf_today_date}")
                        return None

                    if market == 'pre_market':
                        if yf_prev_date not in data.index or yf_today_date not in data.index:
                            logger.warning(f"Missing price data for {ticker} on {yf_prev_date} or {yf_today_date}")
                            return None
                        begin_price = float(data.loc[yf_prev_date, 'Close'])
                        end_price = float(data.loc[yf_today_date, 'Open'])
                        index_begin_price = float(index_data.loc[yf_prev_date, 'Close'])
                        index_end_price = float(index_data.loc[yf_today_date, 'Open'])
                    elif market == 'regular_market':
                        if yf_today_date not in data.index:
                            logger.warning(f"Missing price data for {ticker} on {yf_today_date}")
                            return None
                        begin_price = float(data.loc[yf_today_date, 'Open'])
                        end_price = float(data.loc[yf_today_date, 'Close'])
                        index_begin_price = float(index_data.loc[yf_today_date, 'Open'])
                        index_end_price = float(index_data.loc[yf_today_date, 'Close'])
                    else:  # after_market
                        if yf_today_date not in data.index or yf_next_date not in data.index:
                            logger.warning(f"Missing price data for {ticker} on {yf_today_date} or {yf_next_date}")
                            return None
                        begin_price = float(data.loc[yf_today_date, 'Close'])
                        end_price = float(data.loc[yf_next_date, 'Open'])
                        index_begin_price = float(index_data.loc[yf_today_date, 'Close'])
                        index_end_price = float(index_data.loc[yf_next_date, 'Open'])

                    price_change = end_price - begin_price
                    index_price_change = index_end_price - index_begin_price

                    price_change_percentage = (price_change / begin_price) * 100 if begin_price != 0 else 0
                    index_price_change_percentage = (index_price_change / index_begin_price) * 100 if index_begin_price != 0 else 0

                    volume = float(data.loc[yf_today_date, 'Volume']) if yf_today_date in data.index else 0

                    return {
                        'news_id': news_id,
                        'ticker': ticker,
                        'published_date': published_date,
                        'begin_price': begin_price,
                        'end_price': end_price,
                        'index_begin_price': index_begin_price,
                        'index_end_price': index_end_price,
                        'price_change': price_change,
                        'price_change_percentage': price_change_percentage,
                        'index_price_change': index_price_change,
                        'index_price_change_percentage': index_price_change_percentage,
                        'daily_alpha': price_change_percentage - index_price_change_percentage,
                        'actual_side': 'UP' if price_change_percentage >= 0 else 'DOWN',
                        'volume': volume,
                        'market': market
                    }
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                    if attempt < retries - 1:
                        time_module.sleep(0.5)
                    continue
            logger.error(f"Failed to get price data for {ticker} after {retries} attempts")
            return None

        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {e}")
            return None

    def extract_ticker_from_company(self, company_name: str, news_text: str = "") -> Optional[str]:
        """Extract ticker symbol from company name using OpenAI."""
        if not self.openai_client:
            logger.warning("OpenAI client not initialized - skipping ticker extraction")
            return None

        try:
            if not company_name:
                logger.warning("No company name provided for ticker extraction")
                return None

            text = f"Company: {company_name}"
            if news_text:
                text += f" News: {news_text[:500]}"

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract publicly traded companies from news text. Return JSON with 'tickers' and 'companies' arrays. Example: {\"tickers\": [\"AAPL\", \"MSFT\"], \"companies\": [\"Apple Inc\", \"Microsoft Corporation\"]}. If no companies found, return {\"tickers\": [], \"companies\": []}."
                    },
                    {
                        "role": "user",
                        "content": f"Extract company tickers and names from this text: {text}"
                    }
                ],
                max_tokens=100,
                temperature=0
            )

            result = response.choices[0].message.content
            try:
                parsed = json.loads(result)
                tickers = parsed.get('tickers', [])
                ticker = tickers[0] if tickers else None
                if ticker and self.validate_ticker(ticker):
                    logger.info(f"Valid ticker extracted: {ticker} for {company_name}")
                    return ticker
                logger.warning(f"Invalid or no ticker extracted for {company_name}")
                return None
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse OpenAI response: {result}")
                return None

        except Exception as e:
            logger.error(f"Error extracting ticker for {company_name}: {e}")
            return None

    def save_checkpoint(self, df: pd.DataFrame, processed_df: pd.DataFrame, current_stage: str):
        """Save processing progress to a checkpoint file."""
        checkpoint_data = {
            'original_df': df,
            'processed_df': processed_df,
            'current_stage': current_stage,
            'processed_count': self.processed_count,
            'total_to_process': self.total_to_process
        }
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved to {self.checkpoint_file}. Progress: {self.get_progress_percentage()}%")

    def load_checkpoint(self) -> Optional[dict]:
        """Load processing progress from checkpoint file."""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                self.processed_count = checkpoint_data['processed_count']
                self.total_to_process = checkpoint_data['total_to_process']
                logger.info(f"Resuming from checkpoint. Progress: {self.get_progress_percentage()}%")
                return checkpoint_data
            return None
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def get_progress_percentage(self) -> float:
        """Calculate current progress percentage."""
        if self.total_to_process == 0:
            return 0.0
        return round((self.processed_count / self.total_to_process) * 100, 2)

    def process_data(self, input_file: str, output_file: str, test_size: int = None, openai_api_key: str = None):
        """Process the input CSV file and save results to output file."""
        # Read the input file
        df = pd.read_csv(input_file)
        
        # If test_size is provided, use a sample of the data
        if test_size is not None:
            df = df.sample(min(test_size, len(df)), random_state=42)
            logger.info(f"Sampled {len(df)} rows for processing")

        # Check for existing checkpoint
        checkpoint = self.load_checkpoint()
        if checkpoint:
            df = checkpoint['original_df']
            valid_final_df = checkpoint['processed_df']
            current_stage = checkpoint['current_stage']
        else:
            valid_final_df = None
            current_stage = 'initial'

        # Step 1: Clean up ticker columns
        if current_stage == 'initial':
            df['yf_ticker'] = df['yf_ticker'].fillna(df.get('ticker', ''))
            df['yf_ticker'] = df['yf_ticker'].replace('', pd.NA)
            df = df.drop(columns=['ticker'], errors='ignore')

            # Convert published_date to datetime
            try:
                df['published_date'] = pd.to_datetime(df['published_date'], format='ISO8601', utc=True)
            except Exception as e:
                logger.warning(f"ISO8601 parsing failed, trying mixed format: {e}")
                try:
                    df['published_date'] = pd.to_datetime(df['published_date'], format='mixed', utc=True)
                except Exception as e:
                    logger.error(f"Failed to parse dates after multiple attempts: {e}")
                    raise

            # Separate rows with valid and missing yf_ticker
            valid_ticker_df = df[df['yf_ticker'].notna()].copy()
            missing_ticker_df = df[df['yf_ticker'].isna()].copy()
            logger.info(f"Processing {len(valid_ticker_df)} rows with valid tickers")
            logger.info(f"Processing {len(missing_ticker_df)} rows with missing tickers")

            self.total_to_process = len(valid_ticker_df) + len(missing_ticker_df)
            current_stage = 'processing_valid_tickers'
            self.save_checkpoint(df, None, current_stage)

        # Step 2: Process rows with valid yf_ticker
        if current_stage == 'processing_valid_tickers':
            if valid_final_df is None:
                valid_ticker_df = df[df['yf_ticker'].notna()].copy()
                price_data_list = []

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self.get_price_data, row['yf_ticker'], row['published_date'], row['news_id']): row for _, row in valid_ticker_df.iterrows()}
                    for future in as_completed(futures):
                        self.processed_count += 1
                        price_data = future.result()
                        if price_data:
                            price_data_list.append(price_data)

                        if self.processed_count % self.intermediate_csv_interval == 0:
                            temp_df = pd.DataFrame(price_data_list)
                            if not temp_df.empty:
                                temp_df = valid_ticker_df.merge(temp_df, on='news_id', how='left')
                                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                                temp_df.to_csv(self.output_file, mode='a', index=False, header=not os.path.exists(self.output_file))
                                logger.info(f"Intermediate results saved to {self.output_file} ({self.processed_count} rows processed)")
                                price_data_list = []  # Clear list to avoid duplicates

                        if self.processed_count % self.checkpoint_interval == 0:
                            logger.info(f"Progress: {self.get_progress_percentage()}%")
                            self.save_checkpoint(df, None, current_stage)

                # Save any remaining data
                if price_data_list:
                    temp_df = pd.DataFrame(price_data_list)
                    if not temp_df.empty:
                        temp_df = valid_ticker_df.merge(temp_df, on='news_id', how='left')
                        temp_df.to_csv(self.output_file, mode='a', index=False, header=not os.path.exists(self.output_file))
                        logger.info(f"Intermediate results saved to {self.output_file} ({self.processed_count} rows processed)")
                
                valid_results_df = pd.DataFrame(price_data_list)
                valid_final_df = valid_ticker_df.merge(valid_results_df, on='news_id', how='left')
                current_stage = 'processing_missing_tickers'
                self.save_checkpoint(df, valid_final_df, current_stage)

        # Step 3: Process rows with missing yf_ticker (if OpenAI is available)
        if current_stage == 'processing_missing_tickers':
            missing_ticker_df = df[df['yf_ticker'].isna()].copy()

            if len(missing_ticker_df) > 0 and openai_api_key:
                logger.info("Attempting to extract missing tickers using OpenAI")
                self.openai_client = openai.OpenAI(api_key=openai_api_key)

                extracted_tickers = []
                for _, row in missing_ticker_df.iterrows():
                    ticker = self.extract_ticker_from_company(row['company'], row.get('content', ''))
                    extracted_tickers.append(ticker)
                    self.processed_count += 1

                    if self.processed_count % self.checkpoint_interval == 0:
                        logger.info(f"Progress: {self.get_progress_percentage()}%")
                        self.save_checkpoint(df, valid_final_df, current_stage)

                missing_ticker_df['extracted_yf_ticker'] = extracted_tickers

                # Process rows where we successfully extracted a ticker
                extracted_valid_df = missing_ticker_df[missing_ticker_df['extracted_yf_ticker'].notna()].copy()
                extracted_invalid_df = missing_ticker_df[missing_ticker_df['extracted_yf_ticker'].isna()].copy()
                logger.info(f"Extracted {len(extracted_valid_df)} valid tickers from missing data")

                price_data_list = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self.get_price_data, row['extracted_yf_ticker'], row['published_date'], row['news_id']): row for _, row in extracted_valid_df.iterrows()}
                    for future in as_completed(futures):
                        self.processed_count += 1
                        price_data = future.result()
                        if price_data:
                            price_data_list.append(price_data)

                        if self.processed_count % self.intermediate_csv_interval == 0:
                            temp_df = pd.DataFrame(price_data_list)
                            if not temp_df.empty:
                                temp_df = extracted_valid_df.merge(temp_df, on='news_id', how='left')
                                temp_df.to_csv(self.output_file, mode='a', index=False, header=not os.path.exists(self.output_file))
                                logger.info(f"Intermediate results saved to {self.output_file} ({self.processed_count} rows processed)")
                                price_data_list = []  # Clear list to avoid duplicates

                        if self.processed_count % self.checkpoint_interval == 0:
                            logger.info(f"Progress: {self.get_progress_percentage()}%")
                            self.save_checkpoint(df, valid_final_df, current_stage)

                if price_data_list:
                    extracted_results_df = pd.DataFrame(price_data_list)
                    extracted_final_df = extracted_valid_df.merge(extracted_results_df, on='news_id', how='left')
                    extracted_final_df['yf_ticker'] = extracted_final_df['extracted_yf_ticker']
                    # Save remaining extracted data
                    extracted_final_df.to_csv(self.output_file, mode='a', index=False, header=not os.path.exists(self.output_file))
                    logger.info(f"Intermediate results saved to {self.output_file} ({self.processed_count} rows processed)")
                    final_df = pd.concat([valid_final_df, extracted_final_df, extracted_invalid_df], ignore_index=True)
                else:
                    final_df = pd.concat([valid_final_df, missing_ticker_df], ignore_index=True)
            else:
                final_df = pd.concat([valid_final_df, missing_ticker_df], ignore_index=True)

        # Save final results
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        final_df.to_csv(self.output_file, mode='w', index=False)
        logger.info(f"Processing complete. Final results saved to {self.output_file}")

        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("Checkpoint file removed")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data/backfilling/remaining_news.csv"
    OUTPUT_FILE = "data/backfilling/processed_results_remaining.csv"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TEST_SIZE = 30000
    MAX_WORKERS = 10

    # Initialize and run processor
    processor = PriceDataProcessor(
        openai_api_key=OPENAI_API_KEY,
        max_workers=MAX_WORKERS
    )
    processor.process_data(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        test_size=TEST_SIZE,
        openai_api_key=OPENAI_API_KEY
    )