import numpy as np
import pandas as pd
import pickle
import json
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import logging

# # Below code block is for production use
# # -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Set up MLflow tracking for DagsHub
mlflow.set_tracking_uri('https://dagshub.com/anasalam9692/Airline-Ticket-Price-Predictor.mlflow')
dagshub.init(repo_owner='anasalam9692', repo_name='Airline-Ticket-Price-Predictor', mlflow=True)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the regression model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics_dict = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
        logging.info('Regression metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_df = load_data('./data/processed/test_processed.csv')

            X_test = test_df.drop(columns=['Fare']).values
            y_test = test_df['Fare'].values

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(clf, 'get_params'):
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(clf, "model")

            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error(f'Model evaluation failed: {e}')
            print(f"Error: {e}")


if __name__ == '__main__':
    main()

# def main():
#     mlflow.set_experiment("airline-ticket-price-predictor")
#     with mlflow.start_run() as run:
#         try:
#             clf = load_model('./models/model.pkl')
#             test_data = load_data('./data/processed/test_processed.csv')
#             y_test = load_data('./data/processed/y_test.csv')

#             X_test = test_data.values
#             y_test = y_test.values.ravel()

#             metrics = evaluate_model(clf, X_test, y_test)

#             save_metrics(metrics, 'reports/metrics.json')

#             for metric_name, metric_value in metrics.items():
#                 mlflow.log_metric(metric_name, metric_value)

#             if hasattr(clf, 'get_params'):
#                 params = clf.get_params()
#                 for param_name, param_value in params.items():
#                     mlflow.log_param(param_name, param_value)

#             mlflow.sklearn.log_model(clf, "model")

#             save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

#             mlflow.log_artifact('reports/metrics.json')

#         except Exception as e:
#             logging.error('Failed to complete the model evaluation process: %s', e)
#             print(f"Error: {e}")

# if __name__ == '__main__':
#     main()
