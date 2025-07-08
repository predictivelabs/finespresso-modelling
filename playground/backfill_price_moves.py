import pandas as pd
import yfinance as yf
import json
import time as time_module
from datetime import datetime, time
import logging
from typing import Optional, Dict, List
import openai
import os
import math
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceDataProcessor:
    def __init__(self, openai_api_key: str = None, index_symbol: str = 'SPY'):
        self.index_symbol = index_symbol
        self.openai_api_key = openai_api_key  # Store API key but don't initialize client yet
        self.processed_count = 0
        self.total_to_process = 0
        self.checkpoint_interval = 100
        self.checkpoint_file = "processing_checkpoint.pkl"

    def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid by trying to get info from yfinance."""
        try:
            base_ticker = ticker.split('.')[0]
            info = yf.Ticker(base_ticker).info
            return info is not None and 'symbol' in info
        except:
            return False

    @staticmethod
    def get_price_data_worker(args):
        """Static method for worker processes to avoid pickling issues."""
        ticker, published_date, news_id, index_symbol = args
        try:
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

            # Helper functions for trading days
            def get_previous_trading_day(date):
                date = date - pd.Timedelta(days=1)
                while date.weekday() >= 5:
                    date = date - pd.Timedelta(days=1)
                return date

            def get_next_trading_day(date):
                date = date + pd.Timedelta(days=1)
                while date.weekday() >= 5:
                    date = date + pd.Timedelta(days=1)
                return date

            previous_trading_day = get_previous_trading_day(pub_date)
            next_trading_day = get_next_trading_day(pub_date)

            yf_prev_date = previous_trading_day.strftime('%Y-%m-%d')
            yf_today_date = pub_date.strftime('%Y-%m-%d')
            yf_next_date = next_trading_day.strftime('%Y-%m-%d')

            for attempt in range(3):  # 3 retries
                try:
                    data = yf.download(ticker, start=yf_prev_date, end=yf_next_date, interval='1d', auto_adjust=False)
                    index_data = yf.download(index_symbol, start=yf_prev_date, end=yf_next_date, interval='1d', auto_adjust=False)

                    if data.empty or index_data.empty:
                        return None

                    if market == 'pre_market':
                        if yf_prev_date not in data.index or yf_today_date not in data.index:
                            return None
                        begin_price = float(data.loc[yf_prev_date, 'Close'])
                        end_price = float(data.loc[yf_today_date, 'Open'])
                        index_begin_price = float(index_data.loc[yf_prev_date, 'Close'])
                        index_end_price = float(index_data.loc[yf_today_date, 'Open'])
                    elif market == 'regular_market':
                        if yf_today_date not in data.index:
                            return None
                        begin_price = float(data.loc[yf_today_date, 'Open'])
                        end_price = float(data.loc[yf_today_date, 'Close'])
                        index_begin_price = float(index_data.loc[yf_today_date, 'Open'])
                        index_end_price = float(index_data.loc[yf_today_date, 'Close'])
                    else:  # after_market
                        if yf_today_date not in data.index or yf_next_date not in data.index:
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
                    time_module.sleep(0.5)
                    continue
            return None
        except Exception as e:
            return None

    def extract_ticker_from_company(self, company_name: str, news_text: str = "") -> Optional[str]:
        """Extract ticker symbol from company name using OpenAI."""
        if not self.openai_api_key:
            return None

        try:
            client = openai.OpenAI(api_key=self.openai_api_key)  # Create new client for each call
            if not company_name:
                return None

            text = f"Company: {company_name}"
            if news_text:
                text += f" News: {news_text[:500]}"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract publicly traded companies from news text. Return JSON with 'tickers' and 'companies' arrays."
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
                    return ticker
                return None
            except json.JSONDecodeError:
                return None
        except Exception as e:
            return None

    def process_data(self, input_file: str, output_file: str, test_size: int = None, openai_api_key: str = None):
        """Process the input CSV file and save results to output file."""
        self.openai_api_key = openai_api_key or self.openai_api_key
        
        df = pd.read_csv(input_file)
        if test_size is not None:
            df = df.sample(min(test_size, len(df)), random_state=42)
        
        # Clean up data
        df['yf_ticker'] = df['yf_ticker'].fillna(df.get('ticker', ''))
        df['yf_ticker'] = df['yf_ticker'].replace('', pd.NA)
        df = df.drop(columns=['ticker'], errors='ignore')
        
        try:
            df['published_date'] = pd.to_datetime(df['published_date'], utc=True)
        except Exception as e:
            try:
                df['published_date'] = pd.to_datetime(df['published_date'], format='ISO8601')
            except:
                df['published_date'] = pd.to_datetime(df['published_date'], format='mixed')

        # Separate rows with valid and missing yf_ticker
        valid_ticker_df = df[df['yf_ticker'].notna()].copy()
        missing_ticker_df = df[df['yf_ticker'].isna()].copy()
        
        self.total_to_process = len(valid_ticker_df) + len(missing_ticker_df)
        
        # Process valid tickers in parallel
        logger.info("Processing valid tickers...")
        valid_args = [(row['yf_ticker'], row['published_date'], row['news_id'], self.index_symbol) 
                     for _, row in valid_ticker_df.iterrows()]
        
        with Pool(min(cpu_count(), 8)) as pool:
            results = list(pool.imap_unordered(self.get_price_data_worker, valid_args, chunksize=100))
        
        valid_results = [r for r in results if r is not None]
        valid_results_df = pd.DataFrame(valid_results)
        valid_final_df = valid_ticker_df.merge(valid_results_df, on='news_id', how='left')
        self.processed_count = len(valid_ticker_df)
        
        # Process missing tickers if OpenAI key is available
        if not missing_ticker_df.empty and self.openai_api_key:
            logger.info("Processing missing tickers with OpenAI...")
            
            # Extract tickers first
            def extract_wrapper(row):
                return self.extract_ticker_from_company(row['company'], row.get('content', ''))
            
            missing_ticker_df['extracted_yf_ticker'] = missing_ticker_df.apply(extract_wrapper, axis=1)
            extracted_valid_df = missing_ticker_df[missing_ticker_df['extracted_yf_ticker'].notna()].copy()
            
            # Process extracted tickers in parallel
            if not extracted_valid_df.empty:
                extracted_args = [(row['extracted_yf_ticker'], row['published_date'], row['news_id'], self.index_symbol)
                               for _, row in extracted_valid_df.iterrows()]
                
                with Pool(min(cpu_count(), 8)) as pool:
                    extracted_results = list(pool.imap_unordered(self.get_price_data_worker, extracted_args, chunksize=100))
                
                extracted_valid_results = [r for r in extracted_results if r is not None]
                if extracted_valid_results:
                    extracted_results_df = pd.DataFrame(extracted_valid_results)
                    extracted_final_df = extracted_valid_df.merge(extracted_results_df, on='news_id', how='left')
                    extracted_final_df['yf_ticker'] = extracted_final_df['extracted_yf_ticker']
                    final_df = pd.concat([valid_final_df, extracted_final_df], ignore_index=True)
                else:
                    final_df = pd.concat([valid_final_df, missing_ticker_df], ignore_index=True)
            else:
                final_df = pd.concat([valid_final_df, missing_ticker_df], ignore_index=True)
        else:
            final_df = pd.concat([valid_final_df, missing_ticker_df], ignore_index=True)
        
        final_df.to_csv(output_file, index=False)
        logger.info(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    INPUT_FILE = "data/backfilling/all_news_20250708_114929.csv"
    OUTPUT_FILE = "data/backfilling/processed_results.csv"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TEST_SIZE = 10000  # Set to None to process all rows
    
    processor = PriceDataProcessor(openai_api_key=OPENAI_API_KEY)
    processor.process_data(INPUT_FILE, OUTPUT_FILE, test_size=TEST_SIZE, openai_api_key=OPENAI_API_KEY)