# data_preprocessing.py

import pandas as pd
import os
from sklearn.impute import SimpleImputer
from src.logger import logging

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    null_report = df.isnull().sum()

    # Log null counts
    for col, nulls in null_report.items():
        if nulls > 0:
            logging.info(f"Column '{col}' has {nulls} missing values.")

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Impute numeric columns with mean
    if num_cols:
        imputer_num = SimpleImputer(strategy='mean')
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Impute categorical columns with most frequent
    if cat_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    logging.info("Missing value imputation completed.")
    return df

def main():
    try:
        # Load raw data
        train_df = pd.read_csv('./data/raw/train.csv')
        test_df = pd.read_csv('./data/raw/test.csv')
        logging.info("Raw data loaded successfully.")

        # Impute missing values if needed
        train_cleaned = impute_missing_values(train_df)
        test_cleaned = impute_missing_values(test_df)

        # Save cleaned data to interim
        interim_path = './data/interim'
        os.makedirs(interim_path, exist_ok=True)

        train_cleaned.to_csv(os.path.join(interim_path, 'train_cleaned.csv'), index=False)
        test_cleaned.to_csv(os.path.join(interim_path, 'test_cleaned.csv'), index=False)
        logging.info("Cleaned data saved to interim folder.")

    except Exception as e:
        logging.error("Error during preprocessing: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
