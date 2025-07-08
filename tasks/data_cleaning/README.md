# Data Cleaning Module

This directory contains a comprehensive data cleaning and quality management system for financial news impact prediction data. The module provides tools for data cleaning, validation, metrics monitoring, and versioning, tailored to the financial news dataset with 32 columns: `news_id`, `ticker_url`, `title`, `content`, `link`, `company`, `event`, `reason`, `publisher`, `industry`, `publisher_topic`, `instrument_id`, `yf_ticker`, `published_date`, `published_date_gmt`, `timezone`, `publisher_summary`, `predicted_side`, `predicted_move`, `language`, `begin_price`, `end_price`, `index_begin_price`, `index_end_price`, `price_change`, `price_change_percentage`, `index_price_change`, `index_price_change_percentage`, `daily_alpha`, `actual_side`, `volume`, and `market`.

## Overview

The data cleaning module is designed to handle financial news data with event-specific cleaning strategies, ensuring data quality while preserving important patterns for machine learning models in the **Finespresso Modelling** project.

## Files Description

### 1. `data_cleaner.py` - Main Data Cleaning Engine
**Purpose**: Core data cleaning functionality with event-specific handling for financial news data.

**Key Features**:
- Event-specific outlier detection with different thresholds for event types (e.g., `changes_in_companys_own_shares`, `other`)
- Missing value imputation using event-specific medians for numerical columns (`price_change_percentage`, `daily_alpha`, etc.)
- KNN imputation for numerical columns with high missing rates (>90%)
- Mode imputation for categorical columns (`event`, `language`, `publisher`, `actual_side`, `market`)
- Text cleaning with financial-specific noise removal and multilingual support (English, French, Norwegian)
- Datetime standardization for `published_date`
- Relaxed winsorizing (5% limits) to preserve variance
- Drops `extracted_yf_ticker` column (if present)
- Removes rows missing both `actual_side` and `price_change_percentage`
- Cleans `event` column by imputing or dropping long-text values (e.g., "Without specific content")
- Comprehensive logging and metrics tracking

**Main Class**: `DataCleaner`

**Key Methods**:
- `load_data()`: Load CSV data from `/data/backfilling/aall_news_cleaned__YYYYMMDD_HHMMSS.csv.csv`
- `impute_high_missing_columns()`: Impute columns with >90% missing values using KNN for numerical and mode for categorical
- `impute_missing_values()`: Event-specific imputation for remaining numerical columns
- `detect_outliers_iqr()`: Event-specific outlier detection for `price_change_percentage` and `daily_alpha`
- `handle_outliers()`: Winsorizing with relaxed 5% limits
- `clean_datetime()`: Standardize `published_date`
- `clean_text_data()`: Clean `title` and `content` with financial noise removal
- `clean()`: Run complete cleaning pipeline, saving to `C:/Users/HP/Desktop/Upwork_project/finespresso-modelling/data/clean/cleaned_price_moves_YYYYMMDD.csv`

**Usage**:
```bash
python tasks/data_cleaning/data_cleaner.py
```

### 2. `data_metrics.py` - Data Quality Metrics Monitoring
**Purpose**: Continuous monitoring of data quality metrics with baseline comparison for the validated dataset.

**Key Features**:
- Completeness metrics for all 32 columns (e.g., `news_id`, `ticker_url`, `price_change_percentage`)
- Consistency metrics (unique value counts for `event`, `language`, `publisher`, `actual_side`, `market`, etc.)
- Validity metrics (invalid value detection for `begin_price`, `end_price`, `volume`)
- Outlier detection using IQR for `price_change_percentage`, `daily_alpha`, `index_price_change_percentage`
- Distribution plotting for numerical columns (`begin_price`, `end_price`, `price_change`, etc.)
- Baseline comparison with `baseline_metrics.csv` (optional)
- Automated metrics saving with timestamps
- Enhanced error handling for missing input files

**Main Class**: `DataMetricsMonitor`

**Key Methods**:
- `calculate_completeness()`: Calculate data completeness
- `calculate_consistency()`: Calculate unique value counts for categorical columns
- `calculate_validity()`: Detect invalid (negative) values in numerical columns
- `calculate_outliers()`: Detect outliers using IQR
- `plot_distributions()`: Generate distribution plots for numerical columns
- `compare_with_baseline()`: Compare with baseline metrics from `data/quality_metrics/baseline_metrics.csv`
- `monitor()`: Run complete metrics pipeline, saving to `data/quality_metrics/metrics_YYYYMMDD_HHMMSS.csv`

**Usage**:
```bash
python tasks/data_cleaning/data_metrics.py
```

### 3. `data_validation.py` - Data Validation Engine
**Purpose**: Comprehensive data validation with business rule enforcement for the cleaned dataset.

**Key Features**:
- Input structure validation for 32 expected columns (e.g., `news_id`, `ticker_url`, `published_date`)
- Data type validation and conversion (e.g., `published_date` to `datetime64[ns, UTC]`, `price_change_percentage` to `float`)
- Categorical value balancing for `actual_side` (UP/DOWN) using upsampling
- Relaxed outlier detection (3.0 IQR multiplier) for `price_change_percentage` and `daily_alpha`
- Price range validation (e.g., `price_change_percentage` ≤ 100%)
- Text quality validation for `title` and `content` (minimum length 10 characters)
- Comprehensive error handling for missing files and invalid data
- Outputs validated data to `data/clean/validated_price_moves_YYYYMMDD.csv`

**Main Class**: `DataValidator`

**Key Methods**:
- `check_input_structure()`: Validate input data structure
- `validate_data_types()`: Validate and convert data types
- `validate_categorical_values()`: Balance `actual_side` classes
- `validate_outliers()`: Detect and cap outliers
- `validate_price_ranges()`: Validate `price_change_percentage` and `daily_alpha`
- `validate_text_quality()`: Ensure `title` and `content` quality
- `validate()`: Run complete validation pipeline, saving to `/data/clean/validated_price_moves_YYYYMMDD.csv`

**Usage**:
```bash
python tasks/data_cleaning/data_validation.py
```

### 4. `data_versioning.py` - Data Versioning and Lineage
**Purpose**: Manage dataset versions and track data lineage for reproducibility.

**Key Features**:
- Automatic semantic versioning (e.g., `v1.0.0`, `v1.0.1`)
- SHA256 hash computation for data integrity
- Lineage tracking with processing steps and parameters
- Version manifest management
- Comprehensive audit trail
- Uses `data/clean/cleaned_price_moves_YYYYMMDD.csv` as input and `data/clean/validated_price_moves_YYYYMMDD.csv` as processed data
- Enhanced error handling for missing files

**Main Class**: `DataVersioning`

**Key Methods**:
- `compute_hash()`: Compute SHA256 hash of DataFrame
- `get_next_version()`: Determine next version number
- `save_version()`: Save versioned data to `/data/versions/vX.Y.Z/`
- `save_lineage()`: Save lineage JSON to `C/data/lineage/`
- `update_manifest()`: Update `/data/versions/versions.csv`
- `version_and_track()`: Complete versioning and tracking

**Usage**:
```bash
python tasks/data_cleaning/data_versioning.py
```

### 5. `data_quality_pipeline.py` - Complete Pipeline Orchestrator
**Purpose**: Orchestrate the complete data quality pipeline.

**Key Features**:
- Sequential execution of cleaning, validation, metrics monitoring, and versioning
- Comprehensive logging throughout the pipeline
- Integration of all components (`DataCleaner`, `DataValidator`, `DataMetricsMonitor`, `DataVersioning`)
- Centralized configuration management
- Parameterized processing steps and settings
- Robust error handling for missing files and processing errors
- Uses date-only (`YYYYMMDD`) naming for cleaned and validated outputs

**Main Function**: `run_data_quality_pipeline()`

**Pipeline Steps**:
1. Data Cleaning (using `DataCleaner`): Produces `cleaned_price_moves_YYYYMMDD.csv`
2. Data Validation (using `DataValidator`): Produces `validated_price_moves_YYYYMMDD.csv`
3. Metrics Monitoring (using `DataMetricsMonitor`): Analyzes validated data, compares with `baseline_metrics.csv`
4. Data Versioning (using `DataVersioning`): Versions validated data and tracks lineage

**Usage**:
```bash
cd C:/Users/HP/Desktop/Upwork_project/finespresso-modelling
python data_quality_pipeline.py
```

## Running the Complete Pipeline

To run the entire data quality pipeline:

```bash
cd C:/Users/HP/Desktop/Upwork_project/finespresso-modelling
python data_quality_pipeline.py
```

This will:
1. Clean the data from `data/backfilling/all_news_cleaned__YYYYMMDD_HHMMSS.csv` using event-specific strategies (~2,129 rows after cleaning)
2. Validate the cleaned data, ensuring correct data types, balanced classes, and valid ranges
3. Monitor data quality metrics with baseline comparison
4. Version the validated data and track lineage
5. Save all outputs to appropriate directories

## Output Structure

The pipeline generates the following outputs:

```
C:/Users/HP/Desktop/Upwork_project/finespresso-modelling/
├── data/
│   ├── backfilling/
│   │   └── all_news_cleaned__YYYYMMDD_HHMMSS.csv  # Input data
│   ├── clean/
│   │   ├── cleaned_price_moves_YYYYMMDD.csv          # Cleaned data (~2,129 rows, 32 columns)
│   │   └── validated_price_moves_YYYYMMDD.csv        # Validated data
│   ├── quality_metrics/
│   │   ├── cleaning_metrics.csv                     # Cleaning metrics
│   │   ├── validation_metrics.csv                   # Validation metrics
│   │   ├── baseline_metrics.csv                     # Baseline metrics (optional)
│   │   ├── completeness_report_YYYYMMDD_HHMMSS.csv  # Completeness report
│   │   ├── consistency_report_YYYYMMDD_HHMMSS.csv   # Consistency report
│   │   ├── validity_report_YYYYMMDD_HHMMSS.csv      # Validity report
│   │   ├── outliers_report_YYYYMMDD_HHMMSS.csv      # Outliers report
│   │   ├── metrics_YYYYMMDD_HHMMSS.csv              # Timestamped metrics
│   │   └── plots/                                   # Distribution plots
│   ├── versions/
│   │   ├── v1.0.0/                                  # Versioned data
│   │   │   └── validated_price_moves_YYYYMMDD.csv
│   │   ├── v1.0.1/
│   │   │   └── validated_price_moves_YYYYMMDD.csv
│   │   └── versions.csv                             # Version manifest
│   ├── lineage/
│   │   └── lineage_vX.Y.Z_YYYYMMDD_HHMMSS.json      # Lineage tracking
├── tasks/
│   ├── data_cleaning/
│   │   ├── logs/
│   │   │   ├── cleaning.log                        # Cleaning logs
│   │   │   ├── validation.log                      # Validation logs
│   │   │   ├── metrics.log                         # Metrics logs
│   │   │   ├── versioning.log                      # Versioning logs
│   │   │   └── pipeline.log                        # Pipeline logs
```

## Configuration

The pipeline uses default paths but can be customized by modifying the parameters in `data_quality_pipeline.py`:

- **Input data**: `data/backfilling/all_news_cleaned__YYYYMMDD_HHMMSS.csv`
- **Cleaned data**: `data/clean/cleaned_price_moves_YYYYMMDD.csv`
- **Validated data**: `data/clean/validated_price_moves_YYYYMMDD.csv`
- **Metrics directory**: `C:/Users/HP/Desktop/Upwork_project/finespresso-modelling/data/quality_metrics/`
- **Versions directory**: `data/versions/`
- **Lineage directory**: `data/lineage/`

## Dependencies

Required Python packages (see `requirements.txt`):
- pandas
- numpy
- scipy
- scikit-learn
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
- `tasks/data_cleaning/logs/cleaning.log`
- `tasks/data_cleaning/logs/validation.log`
- `tasks/data_cleaning/logs/metrics.log`
- `tasks/data_cleaning/logs/versioning.log`
- `tasks/data_cleaning/logs/pipeline.log`

## Event-Specific Handling

The cleaning system uses different strategies for different event types:

- **changes_in_companys_own_shares**: Less strict outlier detection (1.5 IQR)
- **other**: Standard outlier detection (1.5 IQR)
- **Default**: Standard outlier detection (1.5 IQR)

This ensures that different types of financial events are handled appropriately based on their inherent characteristics.