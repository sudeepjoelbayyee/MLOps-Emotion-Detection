import os
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> dict:
    """Loads configuration parameters from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        return {}

def load_data(train_path: str, test_path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads training and testing datasets."""
    try:
        train_data = pd.read_csv(train_path).fillna('')
        test_data = pd.read_csv(test_path).fillna('')
        logging.info("Data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies Bag of Words transformation."""
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(train_data['content'].values)
    X_test_bow = vectorizer.transform(test_data['content'].values)
    
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = train_data['sentiment'].values
    
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = test_data['sentiment'].values
    
    logging.info("Bag of Words transformation applied.")
    return train_df, test_df

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Saves processed data as CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_bow.csv"), index=False)
        logging.info("Processed data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")

if __name__ == "__main__":
    config = load_config('params.yaml')
    max_features = config.get('feature_engineering', {}).get('max_features', 1000)
    
    train_data, test_data = load_data("./data/interim/train_processed.csv", "./data/interim/test_processed.csv")
    
    if train_data is not None and test_data is not None:
        train_df, test_df = apply_bow(train_data, test_data, max_features)
        save_data(train_df, test_df, os.path.join("data", "processed"))