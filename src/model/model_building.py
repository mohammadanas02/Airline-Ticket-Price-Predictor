# model_building.py

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from src.logger import logging

# ----------------- Best Parameters ---------------------
BEST_PARAMS = {
    'subsample': 0.8,
    'n_estimators': 150,
    'max_depth': 10,
    'learning_rate': 0.05,
    'gamma': 0,
    'colsample_bytree': 0.6,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

def load_data(file_path: str) -> pd.DataFrame:
    """Load processed data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('CSV parse error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
    """Train the XGBoost model with the best parameters."""
    try:
        model = xgb.XGBRegressor(**BEST_PARAMS)
        model.fit(X_train, y_train)
        logging.info('XGBoost model training completed.')
        return model
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model using pickle."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Failed to save model: %s', e)
        raise

def main():
    try:
        # Load processed training data
        train_df = load_data('./data/processed/train_processed.csv')
        X_train = train_df.drop('Fare', axis=1).values
        y_train = train_df['Fare'].values

        # Train model
        model = train_model(X_train, y_train)

        # Save model
        save_model(model, './models/model.pkl')
        
    except Exception as e:
        logging.error('Failed to complete model building: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
