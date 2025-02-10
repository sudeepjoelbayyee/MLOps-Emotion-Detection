import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging


# Set up logger
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Create Console Handler
console_handler = logging.StreamHandler()

# Create Formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(console_handler)


def load_params(param_path: str) -> float:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params['data_ingestion']['test_size']
    except FileNotFoundError:
        logger.error("File not found")
        return 0.2  # Default test size
    except KeyError:
        logger.error("Missing key 'data_ingestion' or 'test_size' in params file.")
        return 0.2
    except yaml.YAMLError:
        logger.error("Failed to parse YAML file.")
        return 0.2

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error reading data from {url}: {e}")
        return pd.DataFrame()

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.empty:
            raise ValueError("Error: DataFrame is empty.")
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        print(f"Error: Missing column {e} in the DataFrame.")
        return pd.DataFrame()
    except ValueError as e:
        print(e)
        return pd.DataFrame()

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Error saving data: {e}")

def main() -> None:
    try:
        test_size = load_params("params.yaml")
        df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
        final_df = process_data(df)
        if final_df.empty:
            raise ValueError("Error: Processed DataFrame is empty, cannot proceed.")
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
