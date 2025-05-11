import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from src.connections import s3_connection

# ------------------ PARAM LOADER ------------------
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

# ------------------ GITHUB DATA LOADER ------------------
def load_data_from_github(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info('Data successfully loaded from GitHub.')
        return df
    except Exception as e:
        logging.error('Failed to load data from GitHub: %s', e)
        raise

# ------------------ S3 DATA LOADER (for later use) ------------------
def load_data_from_s3(bucket: str, file_key: str, access_key: str, secret_key: str) -> pd.DataFrame:
    s3 = s3_connection.s3_operations(bucket, access_key, secret_key)
    df = s3.fetch_file_from_s3(file_key)
    if df is None:
        raise Exception("Data fetch from S3 failed.")
    return df

# ------------------ DATA SAVER ------------------
def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_df.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logging.info("Train and test data saved in %s", raw_data_path)
    except Exception as e:
        logging.error('Error while saving data: %s', e)
        raise

# ------------------ MAIN INGESTION PROCESS ------------------
def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2  # or use load_params('params.yaml') if needed

        # Load data from GitHub
        github_url = 'https://raw.githubusercontent.com/mohammadanas02/Airline-Fare-Predictor/main/sample_data.csv'
        df = load_data_from_github(github_url)

        # Split and save
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')

        logging.info("Data ingestion from GitHub completed.")

        # For future use - S3 ingestion
        """
        # Uncomment when ready to use S3
        bucket = "my-firstmodel-mlopsproj"
        file_key = "sample_data.csv"
        access_key = "your-access-key"
        secret_key = "your-secret-key"
        df = load_data_from_s3(bucket, file_key, access_key, secret_key)
        save_data(df, df, data_path='./data')  # Save as single file or split
        logging.info("Data ingestion from S3 completed.")
        """

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
