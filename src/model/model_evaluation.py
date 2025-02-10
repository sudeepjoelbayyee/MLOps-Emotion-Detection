import numpy as np
import pandas as pd
import pickle
import os
import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Tuple, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path: str) -> Any:
    """Loads a trained model from a file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def load_data(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads test dataset."""
    try:
        test_data = pd.read_csv(test_path)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logging.info("Test data loaded successfully.")
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return None, None

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluates the model and returns performance metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logging.info("Model evaluation completed.")
        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return {}

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Saves evaluation metrics to a JSON file."""
    try:
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved successfully.")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")

if __name__ == "__main__":
    model = load_model('models/model.pkl')
    X_test, y_test = load_data('./data/processed/test_bow.csv')
    
    if model is not None and X_test is not None and y_test is not None:
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, 'metrics.json')
