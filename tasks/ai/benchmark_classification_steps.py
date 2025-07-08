import os
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
import logging
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Ensure directories exist
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
reports_dir = os.path.join(base_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
logs_dir = os.path.join(base_dir, 'tasks', 'ai', 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[
    logging.FileHandler(os.path.join(logs_dir, 'compare_datasets.log')),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Load spacy English model
nlp = spacy.load("en_core_web_sm")

# MLflow setup
mlflow.set_tracking_uri(f"file:///{os.path.join(base_dir, 'mlruns')}")
mlflow.set_experiment("finespresso_dataset_comparison_binary")

def preprocess(text):
    """Preprocess text using spaCy: lemmatize, remove stop words and punctuation."""
    if not isinstance(text, str) or text.strip() == '':
        return ''
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def calculate_directional_metrics(y_true, y_pred):
    """Calculate directional accuracy metrics for UP and DOWN predictions."""
    up_mask = y_true == 1
    down_mask = y_true == 0
    total_up = np.sum(up_mask)
    total_down = np.sum(down_mask)
    correct_up = np.sum((y_true == y_pred) & up_mask)
    correct_down = np.sum((y_true == y_pred) & down_mask)
    up_accuracy = (correct_up / total_up * 100) if total_up > 0 else 0
    down_accuracy = (correct_down / total_down * 100) if total_down > 0 else 0
    total_predictions = len(y_pred)
    up_predictions = np.sum(y_pred == 1)
    down_predictions = np.sum(y_pred == 0)
    up_pred_pct = (up_predictions / total_predictions * 100) if total_predictions > 0 else 0
    down_pred_pct = (down_predictions / total_predictions * 100) if total_predictions > 0 else 0
    return {
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy,
        'total_up': int(total_up),
        'total_down': int(total_down),
        'correct_up': int(correct_up),
        'correct_down': int(correct_down),
        'up_predictions_pct': up_pred_pct,
        'down_predictions_pct': down_pred_pct
    }

def load_data(dataset_name):
    """Load dataset based on name."""
    data_paths = {
        'simple_data': os.path.join(base_dir, 'data', 'backfilling', 'processed_results.csv'),
        'data_quality': os.path.join(base_dir, 'data', 'clean', 'validated_price_moves_20250707.csv'),
        'feature_engineering': os.path.join(base_dir, 'data', 'feature_engineering', 'selected_features_data_20250707.csv')
    }
    csv_path = data_paths.get(dataset_name)
    logger.info(f"Loading {dataset_name} from: {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"{dataset_name} file not found: {csv_path}")
        return pd.DataFrame(), dataset_name
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {dataset_name}")
        return df, dataset_name
    except Exception as e:
        logger.error(f"Error loading {dataset_name}: {str(e)}")
        return pd.DataFrame(), dataset_name

def prepare_features(df, dataset_name):
    """Prepare features: TF-IDF for text and numerical features."""
    df = df.copy()
    
    # Determine available text columns
    text_cols = ['content', 'title']
    available_text_cols = [col for col in text_cols if col in df.columns]
    
    if not available_text_cols:
        logger.error(f"No valid text columns ({text_cols}) found in {dataset_name}")
        raise ValueError(f"No valid text columns found in {dataset_name}")
    
    # Create text_to_process column
    def select_text(row):
        for col in available_text_cols:
            if pd.notna(row[col]) and isinstance(row[col], str) and row[col].strip() != '':
                return row[col]
        logger.warning(f"No valid text in row {row.name} for {dataset_name}")
        return ''
    
    df['text_to_process'] = df.apply(select_text, axis=1)
    df['processed_content'] = df['text_to_process'].apply(preprocess)
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1000)
    X_text = tfidf.fit_transform(df['processed_content'])
    
    # Numerical features (only for feature_engineering)
    numerical_cols = [
        'market_cap', 'float_shares', 'avg_volume', 'beta', 'recent_volume', 
        'float_ratio', 'day_of_week', 'hour', 'combined_sentiment', 
        'prev_news_sentiment', 'volatility', 'days_since_event'
    ]
    available_num_cols = [col for col in numerical_cols if col in df.columns]
    if available_num_cols and dataset_name == 'feature_engineering':
        X_num = df[available_num_cols].fillna(df[available_num_cols].median())
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        X = hstack([X_text, X_num_scaled])
        logger.info(f"Included numerical features for {dataset_name}: {available_num_cols}")
    else:
        X = X_text
        logger.info(f"No numerical features used for {dataset_name}")
    
    return X, tfidf, df

def train_and_save_model_for_event(event, df, dataset_name, model_name, model):
    """Train and evaluate Random Forest for an event."""
    try:
        event_df = df[df['event'] == event].copy()
        logger.info(f"Processing event: {event}, Model: {model_name}, Dataset: {dataset_name}, Samples: {len(event_df)}")
        event_df = event_df[event_df['actual_side'].isin(['UP', 'DOWN'])]
        if len(event_df) < 50:
            logger.warning(f"Not enough data for event {event} in {dataset_name} (samples: {len(event_df)}). Skipping.")
            return None
        
        X, tfidf, event_df = prepare_features(event_df, dataset_name)
        y = event_df['actual_side'].map({'UP': 1, 'DOWN': 0})
        
        if y.isna().all() or len(y.unique()) < 2:
            logger.warning(f"Only one class or no valid labels for event {event} in {dataset_name}. Skipping.")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        if len(np.unique(y_test)) < 2:
            logger.warning(f"Test set for event {event} in {dataset_name} has only one class. Skipping.")
            return None
        
        model.fit(X_train, y_train)
        
        model_filename = os.path.join(models_dir, f'{dataset_name}_{model_name.lower().replace(" ", "_")}_{event.replace(" ", "_").lower()}_classifier_binary.joblib')
        vectorizer_filename = os.path.join(models_dir, f'{dataset_name}_{model_name.lower().replace(" ", "_")}_{event.replace(" ", "_").lower()}_tfidf_vectorizer_binary.joblib')
        # joblib.dump(model, model_filename)
        # joblib.dump(tfidf, vectorizer_filename)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        directional_metrics = calculate_directional_metrics(y_test, y_pred)
        
        # Prepare MLflow signature and input example
        input_example = X_train[:5].toarray() if hasattr(X_train, "toarray") else X_train[:5]
        signature = infer_signature(input_example, model.predict(input_example))
        
        with mlflow.start_run(run_name=f"{dataset_name}_{model_name.lower().replace(' ', '_')}_{event}_model"):
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("event", event)
            mlflow.log_param("model", model_name)
            mlflow.log_param("vectorizer", "TFIDF")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metrics(directional_metrics)
            mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=input_example)
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run logged: {run_id}")
        
        logger.info(f"Model trained: {model_name}, Event: {event}, Dataset: {dataset_name}, Accuracy: {accuracy:.4f}")
        return {
            'model': model_name,
            'dataset': dataset_name,
            'event': event,
            'language': 'en',
            'acc': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'test_sample': len(y_test),
            'training_sample': len(y_train),
            'total_sample': len(event_df),
            'model_filename': model_filename,
            'vectorizer_filename': vectorizer_filename,
            **directional_metrics
        }
    except Exception as e:
        logger.error(f"Error processing event {event} with {model_name} in {dataset_name}: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def train_and_save_all_events_model(df, dataset_name, model_name, model):
    """Train and evaluate Random Forest on all events."""
    try:
        logger.info(f"Training {model_name} on all events, Dataset: {dataset_name}, Samples: {len(df)}")
        df = df[df['actual_side'].isin(['UP', 'DOWN'])].copy()
        if len(df) < 50:
            logger.warning(f"Not enough data for all events in {dataset_name} (samples: {len(df)}). Skipping.")
            return None
        
        X, tfidf, df = prepare_features(df, dataset_name)
        y = df['actual_side'].map({'UP': 1, 'DOWN': 0})
        
        if y.isna().all() or len(y.unique()) < 2:
            logger.warning(f"Only one class or no valid labels for all events in {dataset_name}. Skipping.")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        if len(np.unique(y_test)) < 2:
            logger.warning(f"Test set for all events in {dataset_name} has only one class. Skipping.")
            return None
        
        model.fit(X_train, y_train)
        
        model_filename = os.path.join(models_dir, f'{dataset_name}_{model_name.lower().replace(" ", "_")}_all_events_classifier_binary.joblib')
        vectorizer_filename = os.path.join(models_dir, f'{dataset_name}_{model_name.lower().replace(" ", "_")}_all_events_tfidf_vectorizer_binary.joblib')
        # joblib.dump(model, model_filename)
        # joblib.dump(tfidf, vectorizer_filename)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        directional_metrics = calculate_directional_metrics(y_test, y_pred)
        
        # Prepare MLflow signature and input example
        input_example = X_train[:5].toarray() if hasattr(X_train, "toarray") else X_train[:5]
        signature = infer_signature(input_example, model.predict(input_example))
        
        with mlflow.start_run(run_name=f"{dataset_name}_{model_name.lower().replace(' ', '_')}_all_events_model"):
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("event", "all_events")
            mlflow.log_param("model", model_name)
            mlflow.log_param("vectorizer", "TFIDF")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metrics(directional_metrics)
            mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=input_example)
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run logged: {run_id}")
        
        return {
            'model': model_name,
            'dataset': dataset_name,
            'event': 'all_events',
            'language': 'en',
            'acc': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'test_sample': len(y_test),
            'training_sample': len(y_train),
            'total_sample': len(df),
            'model_filename': model_filename,
            'vectorizer_filename': vectorizer_filename,
            **directional_metrics
        }
    except Exception as e:
        logger.error(f"Error processing all events with {model_name} in {dataset_name}: {str(e)}")
        logger.exception("Detailed traceback:")
        return None

def calculate_percentage_improvement(results_df):
    """Calculate percentage improvement in accuracy between dataset steps."""
    if results_df.empty:
        logger.warning("No results to calculate improvement metrics.")
        return pd.DataFrame()
    
    improvement_data = []
    
    # Pivot results to compare accuracy across datasets for each event
    try:
        pivot_df = results_df.pivot_table(
            values='acc',
            index='event',
            columns='dataset',
            aggfunc='first'
        ).reset_index()
    except Exception as e:
        logger.error(f"Error creating pivot table: {str(e)}")
        return pd.DataFrame()
    
    # Ensure 'event' is a column
    if 'event' not in pivot_df.columns:
        logger.error("Pivot table does not contain 'event' column.")
        return pd.DataFrame()
    
    # Define dataset order for comparison
    dataset_order = ['simple_data', 'data_quality', 'feature_engineering']
    pivot_df = pivot_df.reindex(columns=['event'] + [col for col in dataset_order if col in pivot_df.columns])
    
    # Calculate percentage improvement
    for _, row in pivot_df.iterrows():
        event = row['event']
        acc_simple = row.get('simple_data', np.nan)
        acc_quality = row.get('data_quality', np.nan)
        acc_feature = row.get('feature_engineering', np.nan)
        
        # Step 1 to Step 2: simple_data -> data_quality
        cpercentage_1_to_2 = np.nan
        if not np.isnan(acc_simple) and not np.isnan(acc_quality) and acc_simple > 0:
            cpercentage_1_to_2 = ((acc_quality - acc_simple) / acc_simple) * 100
        
        # Step 2 to Step 3: data_quality -> feature_engineering
        cpercentage_2_to_3 = np.nan
        if not np.isnan(acc_quality) and not np.isnan(acc_feature) and acc_quality > 0:
            cpercentage_2_to_3 = ((acc_feature - acc_quality) / acc_quality) * 100
        
        # Add improvement metrics for each dataset
        for dataset in dataset_order:
            acc = row.get(dataset, np.nan)
            if not np.isnan(acc):
                improvement_data.append({
                    'event': event,
                    'dataset': dataset,
                    'acc': acc,
                    'cpercentage_1_to_2': cpercentage_1_to_2 if dataset == 'data_quality' else np.nan,
                    'cpercentage_2_to_3': cpercentage_2_to_3 if dataset == 'feature_engineering' else np.nan
                })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Log improvement metrics
    for event in improvement_df['event'].unique():
        event_df = improvement_df[improvement_df['event'] == event]
        for _, row in event_df.iterrows():
            dataset = row['dataset']
            acc = row['acc']
            cpercentage_1_to_2 = row['cpercentage_1_to_2']
            cpercentage_2_to_3 = row['cpercentage_2_to_3']
            logger.info(f"Event: {event}, Dataset: {dataset}, Accuracy: {acc:.4f}")
            if not np.isnan(cpercentage_1_to_2):
                logger.info(f"Event: {event}, Improvement (simple_data -> data_quality): {cpercentage_1_to_2:.2f}%")
            if not np.isnan(cpercentage_2_to_3):
                logger.info(f"Event: {event}, Improvement (data_quality -> feature_engineering): {cpercentage_2_to_3:.2f}%")
    
    return improvement_df

def process_results(results):
    """Process and save Random Forest results to a CSV with improvement metrics."""
    try:
        results = [r for r in results if r and 'acc' in r and r['acc'] is not None]
        if not results:
            logger.warning("No valid model results to process.")
            return
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['dataset', 'event', 'acc'], ascending=[True, True, False])
        
        # Calculate percentage improvement
        improvement_df = calculate_percentage_improvement(results_df)
        
        # Merge improvement metrics with results
        if not improvement_df.empty:
            results_df = results_df.merge(
                improvement_df[['event', 'dataset', 'cpercentage_1_to_2', 'cpercentage_2_to_3']],
                on=['event', 'dataset'],
                how='left'
            )
        
        results_csv = os.path.join(reports_dir, 'model_comparison_across_datasets.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Saved model comparison to {results_csv}")
        
    except Exception as e:
        logger.error(f"Error processing results: {str(e)}")
        logger.exception("Detailed traceback:")

def main():
    """Main function to benchmark Random Forest across datasets."""
    logger.info("Starting dataset comparison pipeline")
    
    datasets = ['simple_data', 'data_quality', 'feature_engineering']
    results = []
    
    for dataset_name in datasets:
        df, dataset_name = load_data(dataset_name)
        if df.empty:
            logger.error(f"No data loaded for {dataset_name}, skipping")
            continue
        
        required_cols = ['event', 'actual_side', 'content', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in {dataset_name}: {missing_cols}. Skipping.")
            continue
        
        df = df.dropna(subset=required_cols)
        logger.info(f"Shape after removing null values for {dataset_name}: {df.shape}")
        
        unique_events = df['event'].dropna().unique()
        logger.info(f"Training for {len(unique_events)} events in {dataset_name}: {list(unique_events)}")
        
        model_name = 'Random Forest'
        model = RandomForestClassifier(random_state=42)
        
        # Train per event
        for event in unique_events:
            result = train_and_save_model_for_event(event, df, dataset_name, model_name, model)
            if result is not None:
                results.append(result)
        
        # Train on all events
        combined_result = train_and_save_all_events_model(df, dataset_name, model_name, model)
        if combined_result:
            results.append(combined_result)
    
    process_results(results)
    logger.info("Finished dataset comparison pipeline")

if __name__ == '__main__':
    main()