import sys
import os
import pandas as pd
from datetime import datetime
import logging



# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db.news_db_util import get_news_df


# Configure logging
log_dir = os.path.join('playground', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'get_news.log')

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
logger.info("Logging initialized for get_news")


def get_and_save_news():
    """Fetch news data from database and save to CSV."""
    try:
        # Get timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'data/backfilling'
        output_file = os.path.join(output_dir, f'all_news_{timestamp}.csv')

        # Fetch news data
        logger.info("Fetching news data from database")
        news_df = get_news_df()

        # Check if data was retrieved
        if news_df.empty:
            logger.warning("No news data retrieved from the database")
            print("No news data retrieved from the database.")
            return

        # Save to CSV
        logger.info(f"Saving news data to {output_file}")
        os.makedirs(output_dir, exist_ok=True)
        news_df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(news_df)} rows to {output_file}")
        print(f"Saved news data to {output_file}")

    except Exception as e:
        logger.error(f"Error in get_and_save_news: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    get_and_save_news()