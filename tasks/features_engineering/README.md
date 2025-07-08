# Features Engineering Module

This directory contains a comprehensive feature engineering system for financial news impact prediction data. The module provides tools for data enrichment, feature creation, and feature selection to prepare data for machine learning models in the **Finespresso Modelling** project.

## Overview

The features engineering module transforms cleaned financial news data into rich feature sets that capture market dynamics, temporal patterns, company characteristics, and sentiment information. It integrates with the data quality pipeline, using `validated_price_moves_20250707.csv` as input, and includes external data enrichment, time-based features, company-specific features, and intelligent feature selection.

## Files Description

### 1. `feature_engineering_pipeline.py` - Complete Pipeline Orchestrator
**Purpose**: Orchestrates the entire feature engineering pipeline from data loading to final feature selection.

**Key Features**:
- Sequential execution of all feature engineering steps
- Comprehensive logging and error handling
- Intermediate data saving for debugging
- Integration of all feature engineering components
- Automated pipeline execution

**Main Function**: `run_feature_engineering_pipeline()`

**Pipeline Steps**:
1. **Yahoo Finance Enrichment**: Adds market data and company information
2. **Time Features**: Adds temporal and market timing features
3. **Company Features**: Adds company-specific characteristics
4. **Feature Selection**: Selects optimal feature subset

**Usage**:
```bash
cd <project_root>
python feature_engineering_pipeline.py
```
**Note**: Replace `<project_root>` with the project directory (e.g., `C:/Users/HP/Desktop/Upwork_project/finespresso-modelling`).

### 2. `fetch_yfinance.py` - External Data Enrichment
**Purpose**: Enriches the dataset with Yahoo Finance market data and company information.

**Key Features**:
- Fetches real-time market data using yfinance API
- Implements intelligent caching to reduce API calls
- Retry mechanism with exponential backoff
- Ticker validation and error handling
- Sector performance calculation

**Main Function**: `enrich_with_yfinance()`

**Added Features**:
- `market_cap`: Company market capitalization
- `float_shares`: Number of shares available for trading
- `exchange`: Stock exchange (NYSE, NASDAQ, etc.)
- `sector`: Company sector classification
- `industry`: Company industry classification
- `avg_volume`: Average trading volume
- `beta`: Stock volatility relative to market
- `recent_volume`: Most recent trading volume
- `float_ratio`: Ratio of float shares to market cap
- `sector_performance`: Sector index performance

**Key Methods**:
- `fetch_yf_data()`: Fetch data with retry mechanism
- `validate_ticker()`: Validate ticker symbols
- `load_cache()` / `save_cache()`: Cache management
- `enrich_with_yfinance()`: Main enrichment function

**Usage**:
```bash
cd <project_root>
python tasks/feature_engineering/fetch_yfinance.py
```

### 3. `time_features.py` - Temporal Feature Engineering
**Purpose**: Creates time-based features that capture market timing and temporal patterns.

**Key Features**:
- Day of week and hour extraction
- Weekend and market hours identification
- Earnings season detection
- Quarter-end identification
- Days since event calculation

**Main Function**: `add_time_features()`

**Added Features**:
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `hour`: Hour of day (0-23)
- `is_weekend`: Binary flag for weekends
- `is_market_hours`: Binary flag for market hours (9:30 AM - 4:00 PM EST)
- `is_earnings_season`: Binary flag for earnings season
- `is_quarter_end`: Binary flag for quarter-end months
- `days_since_event`: Days elapsed since news publication

**Market Timing Logic**:
- **Market Hours**: 9:30 AM - 4:00 PM EST (US markets)
- **Earnings Season**: Months following quarter ends (Jan, Apr, Jul, Oct)
- **Quarter End**: March, June, September, December

**Usage**:
```bash
cd <project_root>
python tasks/feature_engineering/time_features.py
```

### 4. `company_features.py` - Company-Specific Features
**Purpose**: Creates company-specific features that capture firm characteristics and historical patterns.

**Key Features**:
- Market cap categorization (Small, Mid, Large)
- Volatility calculation using historical price data
- Sector-relative volatility comparison
- Previous news sentiment analysis
- TextBlob sentiment analysis for news content
- Robust ticker validation and imputation for invalid tickers

**Main Function**: `add_company_features()`

**Added Features**:
- `market_cap_category`: Categorical market cap (Small < $2B, Mid $2B-$10B, Large > $10B)
- `volatility`: Annualized stock volatility (252-day rolling)
- `sector_relative_volatility`: Stock volatility relative to sector index
- `prev_news_sentiment`: Average sentiment of previous news for the company
- `combined_sentiment`: Sentiment score from title + content

**Volatility Calculation**:
- Uses 1-year historical price data
- Annualized using √252 (trading days)
- Sector comparison using sector indices (NASDAQ for Tech, Banking for Finance, etc.)
- Invalid tickers (e.g., `$NQ=F`, `$AZN.ST`) are logged to `data/reports/invalid_tickers_20250707_HHMMSS.csv`
- Missing values (e.g., 216 rows) imputed with median volatility and sector-relative volatility

**Usage**:
```bash
cd <project_root>
python tasks/feature_engineering/company_features.py
```

### 5. `features_selection.py` - Intelligent Feature Selection
**Purpose**: Performs feature selection using correlation analysis and Random Forest importance.

**Key Features**:
- Correlation-based feature elimination
- Random Forest importance-based selection
- Categorical feature encoding (One-Hot Encoding)
- Comprehensive feature validation
- Feature importance reporting
- Outputs only selected features in final CSV

**Main Function**: `select_features()`

**Selection Methods**:
1. **Correlation Filter**: Removes highly correlated features (>0.8 threshold)
2. **Random Forest Importance**: Keeps features with importance > 0.005
3. **Combined Approach**: Applies both filters sequentially

**Parameters**:
- `correlation_threshold`: Maximum correlation between features (default: 0.8)
- `importance_threshold`: Minimum Random Forest importance (default: 0.005)
- `method`: Selection method ('correlation', 'rf', 'correlation_and_rf')

**Preserved Columns**:
- `event`: Event type (always preserved)
- `content`: News content
- `title`: News title
- `actual_side`: Target variable
- `price_change_percentage`: Target variable

**Selected Features** (example output):
- Numerical: `market_cap`, `float_shares`, `avg_volume`, `beta`, `recent_volume`, `float_ratio`, `day_of_week`, `hour`, `combined_sentiment`, `prev_news_sentiment`, `volatility`, `sector_relative_volatility`, `days_since_event`
- Categorical (encoded): `exchange_*`, `sector_*`, `industry_*`, `market_cap_category_*`

**Usage**:
```bash
cd <project_root>
python tasks/feature_engineering/features_selection.py
```

## Running the Complete Pipeline

To run the entire feature engineering pipeline, first ensure the data quality pipeline has generated the input file:

```bash
cd <project_root>
python data_quality_pipeline.py
python feature_engineering_pipeline.py
```

This will:
1. Load cleaned data from `<project_root>/data/clean/validated_price_moves_20250707.csv`
2. Enrich with Yahoo Finance data
3. Add time-based features
4. Add company-specific features
5. Perform feature selection
6. Save final enriched data to `<project_root>/data/feature_engineering/selected_features_data_20250707.csv`

**Note**: Ensure the working directory is set to `<project_root>` (e.g., `C:/Users/HP/Desktop/Upwork_project/finespresso-modelling`) when running `features_selection.py`, as it uses `os.getcwd()` for paths.

## Output Structure

The pipeline generates the following outputs:

```
<project_root>/
├── data/
│   ├── clean/
│   │   └── validated_price_moves_20250707.csv        # Input from data quality pipeline
│   ├── feature_engineering/
│   │   ├── yfinance_enriched_data_20250707.csv       # Data with Yahoo Finance features
│   │   ├── time_features_data_20250707.csv           # Data with time features
│   │   ├── company_features_data_20250707.csv        # Data with company features
│   │   └── selected_features_data_20250707.csv       # Final model-ready dataset
│   ├── cache/
│   │   └── yfinance_cache.pkl                       # Yahoo Finance data cache
│   ├── quality_metrics/
│   │   └── yfinance_metrics_20250707_HHMMSS.csv     # Yahoo Finance fetch metrics
│   ├── features_reports/
│   │   ├── time_feature_stats_20250707_HHMMSS.csv   # Time feature statistics
│   │   ├── company_features_stats_20250707_HHMMSS.csv # Company feature statistics
│   │   ├── feature_importance_20250707_HHMMSS.csv   # Feature importance rankings
│   │   └── invalid_tickers_20250707_HHMMSS.csv      # Invalid tickers from company features
├── tasks/
│   ├── feature_engineering/
│   │   ├── logs/
│   │   │   ├── yfinance.log                        # Yahoo Finance logs
│   │   │   ├── time_features.log                   # Time features logs
│   │   │   ├── company_features.log                # Company features logs
│   │   │   ├── features_selection.log              # Feature selection logs
│   │   │   └── feature_engineering_pipeline.log    # Pipeline logs
```

## Feature Categories

### Market Data Features (Yahoo Finance)
- **Company Information**: Market cap, sector, industry, exchange
- **Trading Metrics**: Volume, beta, float shares
- **Sector Performance**: Relative sector index performance

### Temporal Features
- **Time Components**: Day of week, hour, days since event
- **Market Timing**: Weekend flags, market hours, earnings season
- **Quarterly Patterns**: Quarter-end identification

### Company-Specific Features
- **Size Classification**: Market cap categories
- **Risk Metrics**: Volatility, sector-relative volatility
- **Sentiment History**: Previous news sentiment patterns

### Text Features
- **Sentiment Analysis**: TextBlob sentiment scores
- **Content Features**: Title and content text (preserved for NLP)

## Configuration

Default parameters can be modified in each script:

**Feature Selection Parameters**:
```python
params = {
    'method': 'correlation_and_rf',
    'correlation_threshold': 0.8,
    'importance_threshold': 0.005
}
```

**Market Cap Categories**:
- Small Cap: < $2 billion
- Mid Cap: $2-10 billion
- Large Cap: > $10 billion

**Sector Indices**:
- Technology: ^IXIC (NASDAQ)
- Finance: ^IXBK (Banking)
- Healthcare: ^IXHC (Healthcare)
- Default: ^GSPC (S&P 500)

## Dependencies

Required Python packages (see `requirements.txt`):
- pandas
- numpy
- yfinance
- scikit-learn
- textblob
- retrying
- spacy
- matplotlib
- seaborn
- python-dateutil

Additional setup for multilingual text processing:
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download nb_core_news_sm
```

## Logging

All scripts generate comprehensive logs stored in:
- `<project_root>/tasks/feature_engineering/logs/feature_engineering_pipeline.log`
- `<project_root>/tasks/feature_engineering/logs/yfinance.log`
- `<project_root>/tasks/feature_engineering/logs/time_features.log`
- `<project_root>/tasks/feature_engineering/logs/company_features.log`
- `<project_root>/tasks/feature_engineering/logs/features_selection.log`

## Caching

The Yahoo Finance module implements intelligent caching:
- **Cache Location**: `<project_root>/data/cache/yfinance_cache.pkl`
- **Cache Key**: `{ticker}_{YYYYMMDD}`
- **Cache Benefits**: Reduces API calls, improves performance
- **Cache Management**: Automatic loading/saving with error handling

## Error Handling

The pipeline includes robust error handling:
- **API Failures**: Retry mechanism with exponential backoff
- **Invalid Tickers**: Handled in `company_features.py` with imputation and logging to `data/reports/invalid_tickers_20250707_HHMMSS.csv`
- **Missing Data**: Imputation for numerical and categorical features
- **Feature Dependencies**: Validation of required columns
- **Feature Selection**: Ensures only selected features are saved in `selected_features_data_20250707.csv`

## Performance Considerations

- **API Rate Limiting**: Built-in delays and retry logic
- **Memory Management**: Efficient DataFrame operations
- **Caching**: Reduces redundant API calls
- **Parallel Processing**: Sequential execution for data consistency
- **Portability**: `features_selection.py` uses `os.getcwd()` for dynamic paths, assuming execution from project root