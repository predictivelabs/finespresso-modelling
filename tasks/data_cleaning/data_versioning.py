import pandas as pd
import os
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
import hashlib

def setup_logger(name: str) -> logging.Logger:
    """Configure logging for the module."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, 'tasks', 'data_cleaning', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'versioning.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger(__name__)

class DataVersioning:
    """Class to manage dataset versioning and lineage tracking."""
    
    def __init__(
        self,
        input_path: str = f'data/clean/cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv',
        processed_path: str = f'data/clean/validated_price_moves_{datetime.now().strftime("%Y%m%d")}.csv',
        versions_dir: str = 'data/versions',
        lineage_dir: str = 'data/lineage',
        versions_manifest: str = 'data/versions/versions.csv'
    ):
        """
        Initialize DataVersioning.

        Args:
            input_path (str): Path to cleaned data from DataCleaner.
            processed_path (str): Path to validated data from DataValidator.
            versions_dir (str): Directory for versioned data.
            lineage_dir (str): Directory for lineage logs.
            versions_manifest (str): Path to versions manifest.
        """
        self.input_path = input_path
        self.processed_path = processed_path
        self.versions_dir = versions_dir
        self.lineage_dir = lineage_dir
        self.versions_manifest = versions_manifest
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.lineage_dir, exist_ok=True)

    def compute_hash(self, df: pd.DataFrame) -> str:
        """Compute SHA256 hash of DataFrame."""
        df_str = df.to_csv(index=False).encode()
        return hashlib.sha256(df_str).hexdigest()

    def get_next_version(self) -> str:
        """Determine the next version number."""
        if not os.path.exists(self.versions_manifest):
            return 'v1.0.0'
        versions_df = pd.read_csv(self.versions_manifest)
        latest_version = versions_df['version'].max()
        if not latest_version:
            return 'v1.0.0'
        major, minor, patch = map(int, latest_version[1:].split('.'))
        return f'v{major}.{minor}.{patch + 1}'

    def save_version(self, df: pd.DataFrame, version: str) -> str:
        """Save DataFrame to versioned directory."""
        version_dir = os.path.join(self.versions_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        versioned_path = os.path.join(version_dir, os.path.basename(self.processed_path))
        df.to_csv(versioned_path, index=False)
        logger.info(f"Saved version {version} to {versioned_path}")
        return versioned_path

    def save_lineage(
        self,
        version: str,
        processing_steps: List[str],
        parameters: Dict,
        input_hash: str,
        output_hash: str
    ) -> str:
        """Save lineage information."""
        lineage = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'input_path': self.input_path,
            'input_hash': input_hash,
            'output_path': self.processed_path,
            'output_hash': output_hash,
            'processing_steps': processing_steps,
            'parameters': parameters
        }
        lineage_path = os.path.join(self.lineage_dir, f'lineage_{version}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(lineage_path, 'w') as f:
            json.dump(lineage, f, indent=2)
        logger.info(f"Saved lineage to {lineage_path}")
        return lineage_path

    def update_manifest(
        self,
        version: str,
        versioned_path: str,
        input_hash: str,
        output_hash: str
    ) -> None:
        """Update versions manifest."""
        manifest_data = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'input_path': self.input_path,
            'input_hash': input_hash,
            'output_hash': output_hash,
            'versioned_path': versioned_path
        }
        manifest_df = pd.DataFrame([manifest_data])
        if os.path.exists(self.versions_manifest):
            existing_df = pd.read_csv(self.versions_manifest)
            manifest_df = pd.concat([existing_df, manifest_df], ignore_index=True)
        manifest_df.to_csv(self.versions_manifest, index=False)
        logger.info(f"Updated versions manifest at {self.versions_manifest}")

    def version_and_track(
        self,
        processing_steps: List[str],
        parameters: Dict
    ) -> Tuple[str, str, str]:
        """Version data and track lineage."""
        try:
            input_df = pd.read_csv(self.input_path)
        except Exception as e:
            logger.error(f"Failed to load input data from {self.input_path}: {str(e)}")
            raise ValueError(f"Failed to load {self.input_path}: {str(e)}")
        
        try:
            output_df = pd.read_csv(self.processed_path)
        except Exception as e:
            logger.error(f"Failed to load processed data from {self.processed_path}: {str(e)}")
            raise ValueError(f"Failed to load {self.processed_path}: {str(e)}")
        
        input_hash = self.compute_hash(input_df)
        output_hash = self.compute_hash(output_df)
        version = self.get_next_version()
        versioned_path = self.save_version(output_df, version)
        lineage_path = self.save_lineage(version, processing_steps, parameters, input_hash, output_hash)
        self.update_manifest(version, versioned_path, input_hash, output_hash)
        return version, versioned_path, lineage_path

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    versioning = DataVersioning(
        input_path=os.path.join(data_dir, 'clean', f'cleaned_price_moves_{datetime.now().strftime("%Y%m%d")}.csv'),
        processed_path=os.path.join(data_dir, 'clean', f'validated_price_moves_{datetime.now().strftime("%Y%m%d")}.csv')
    )
    version, versioned_path, lineage_path = versioning.version_and_track(
        processing_steps=[
            'check_input_structure',
            'validate_data_types',
            'validate_categorical_values',
            'validate_price_ranges',
            'validate_outliers',
            'validate_text_quality'
        ],
        parameters={
            'price_change_percentage_threshold': 100,
            'daily_alpha_threshold': 1000,
            'outlier_iqr_multiplier': 3.0,
            'class_balance_threshold': 0.2,
            'min_text_length': 10
        }
    )

