# feature_engineering.py

import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.logger import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Time category mapping
time_categories = {
    'Before 6 AM': 0,
    '6 AM - 12 PM': 1,
    '12 PM - 6 PM': 2,
    'After 6 PM': 3
}

def preprocess_data(df):
    df = df.copy()
    df['Departure_cat'] = df['Departure'].map(time_categories)
    df['Arrival_cat'] = df['Arrival'].map(time_categories)
    df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'], errors='coerce')
    df['Month'] = df['Date_of_journey'].dt.month
    df['Year'] = df['Date_of_journey'].dt.year
    df.drop(['Date_of_journey', 'Departure', 'Arrival'], axis=1, inplace=True, errors='ignore')
    return df

def main():
    try:
        # Load cleaned data
        train_df = pd.read_csv('./data/interim/train_cleaned.csv')
        test_df = pd.read_csv('./data/interim/test_cleaned.csv')
        logging.info("Cleaned train and test data loaded.")

        # Feature Engineering
        train_df = preprocess_data(train_df)
        test_df = preprocess_data(test_df)

        # Drop rows without 'Fare' if any (for completeness)
        train_df = train_df.dropna(subset=['Fare'])

        # Define feature columns
        categorical_features = ['Airline', 'Flight_code', 'Class', 'Source', 'Destination', 'Total_stops', 'Journey_day']
        numerical_features = ['Duration_in_hours', 'Days_left', 'Month', 'Year', 'Departure_cat', 'Arrival_cat']

        # Separate features and target for training
        X_train = train_df.drop('Fare', axis=1)
        y_train = train_df['Fare']

        # Separate test features and retain target if available
        X_test = test_df.drop(columns=['Fare'], errors='ignore')
        y_test = test_df['Fare'] if 'Fare' in test_df.columns else None

        # Define transformers
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer([
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Fit on train and transform both
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Convert to DataFrame with proper column names
        encoded_cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        processed_cols = list(encoded_cat_cols) + numerical_features

        train_processed = pd.DataFrame(X_train_transformed, columns=processed_cols)
        train_processed['Fare'] = y_train.values

        test_processed = pd.DataFrame(X_test_transformed, columns=processed_cols)
        if y_test is not None:
            test_processed['Fare'] = y_test.values

        # Save to CSV
        processed_path = './data/processed'
        os.makedirs(processed_path, exist_ok=True)

        train_processed.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)

        logging.info("Processed data saved to data/processed.")

    except Exception as e:
        logging.error("Error in feature engineering: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()






#__________________________________________________________________________________________________________________
# # feature_engineering.py

# import pandas as pd
# import os
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from src.logger import logging

# # Setup Logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Time category mapping
# time_categories = {
#     'Before 6 AM': 0,
#     '6 AM - 12 PM': 1,
#     '12 PM - 6 PM': 2,
#     'After 6 PM': 3
# }

# def preprocess_data(df):
#     df = df.copy()
#     df['Departure_cat'] = df['Departure'].map(time_categories)
#     df['Arrival_cat'] = df['Arrival'].map(time_categories)
#     df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'], errors='coerce')
#     df['Month'] = df['Date_of_journey'].dt.month
#     df['Year'] = df['Date_of_journey'].dt.year
#     df.drop(['Date_of_journey', 'Departure', 'Arrival'], axis=1, inplace=True, errors='ignore')
#     return df

# def main():
#     try:
#         # Load cleaned data
#         train_df = pd.read_csv('./data/interim/train_cleaned.csv')
#         test_df = pd.read_csv('./data/interim/test_cleaned.csv')
#         logging.info("Cleaned train and test data loaded.")

#         # Feature Engineering
#         train_df = preprocess_data(train_df)
#         test_df = preprocess_data(test_df)

#         # Drop rows without 'Fare' if any (for completeness)
#         train_df = train_df.dropna(subset=['Fare'])

#         # Define feature columns
#         categorical_features = ['Airline', 'Flight_code', 'Class', 'Source', 'Destination', 'Total_stops', 'Journey_day']
#         numerical_features = ['Duration_in_hours', 'Days_left', 'Month', 'Year', 'Departure_cat', 'Arrival_cat']

#         # Separate features and target
#         X_train = train_df.drop('Fare', axis=1)
#         y_train = train_df['Fare']
#         X_test = test_df.copy()

#         # Define transformers
#         categorical_transformer = Pipeline([
#             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#         ])
#         numerical_transformer = Pipeline([
#             ('scaler', StandardScaler())
#         ])
#         preprocessor = ColumnTransformer([
#             ('num', numerical_transformer, numerical_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])

#         # Fit on train and transform both
#         X_train_transformed = preprocessor.fit_transform(X_train)
#         X_test_transformed = preprocessor.transform(X_test)

#         # Convert to DataFrame with proper column names
#         encoded_cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
#         processed_cols = list(encoded_cat_cols) + numerical_features

#         train_processed = pd.DataFrame(X_train_transformed, columns=processed_cols)
#         train_processed['Fare'] = y_train.values

#         test_processed = pd.DataFrame(X_test_transformed, columns=processed_cols)

#         # Save to CSV
#         processed_path = './data/processed'
#         os.makedirs(processed_path, exist_ok=True)

#         train_processed.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
#         test_processed.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)

#         logging.info("Processed data saved to data/processed.")

#     except Exception as e:
#         logging.error("Error in feature engineering: %s", e)
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()
