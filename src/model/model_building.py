import numpy as np
import pandas as pd
import pickle
import os
import logging
import yaml
from sklearn.ensemble import GradientBoostingClassifier
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

def load_data(train_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Loads training dataset."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        logging.info("Training data loaded successfully.")
        return X_train, y_train
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        return None

def train_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, learning_rate: float) -> GradientBoostingClassifier:
    """Trains a Gradient Boosting Classifier."""
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)
    logging.info("Model training completed.")
    return clf

def save_model(model: GradientBoostingClassifier, model_path: str) -> None:
    """Saves the trained model using pickle."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

if __name__ == "__main__":
    config = load_config('params.yaml')
    model_params = config.get('model_building', {})
    n_estimators = model_params.get('n_estimators', 100)
    learning_rate = model_params.get('learning_rate', 0.1)
    
    data = load_data("./data/processed/train_bow.csv")
    
    if data is not None:
        X_train, y_train = data
        model = train_model(X_train, y_train, n_estimators, learning_rate)
        save_model(model, 'models/model.pkl')
