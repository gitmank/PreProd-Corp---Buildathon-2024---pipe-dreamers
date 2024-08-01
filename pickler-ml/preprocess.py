import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load data from a CSV or Excel file.
    
    Parameters:
    file_path (str): Path to the CSV or Excel file.
    
    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")
    
    return data

def preprocess_data(data, remove_features=None, numerical_features=None, categorical_features=None, target_feature='target'):
    """
    Preprocess the data by handling missing values and encoding categorical variables.
    
    Parameters:
    data (pd.DataFrame): Input data as a pandas DataFrame.
    remove_features (list): List of features to remove from the data.
    numerical_features (list): List of numerical features to process.
    categorical_features (list): List of categorical features to process.
    target_feature (str): The target feature for prediction.
    
    Returns:
    pd.DataFrame: Preprocessed data.
    """
    # Remove specified features
    if remove_features:
        data = data.drop(columns=remove_features)
    
    # Separate features and target
    if target_feature in data.columns:
        X = data.drop(columns=[target_feature])
        y = data[target_feature]
    else:
        X = data
        y = None
    
    # Automatically identify numerical and categorical columns if not provided
    if not numerical_features:
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if not categorical_features:
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines for numerical and categorical data
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Convert to DataFrame
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())
    
    if y is not None:
        X_preprocessed[target_feature] = y.values
    
    return X_preprocessed

def save_data(data, file_path):
    """
    Save the preprocessed data to a CSV file.
    
    Parameters:
    data (pd.DataFrame): Data to be saved.
    file_path (str): Path to the output CSV file.
    """
    data.to_csv(file_path, index=False)

def main(input_file, output_file, remove_features=None, numerical_features=None, categorical_features=None, target_feature='target'):
    data = load_data(input_file)
    cleaned_data = preprocess_data(data, remove_features, numerical_features, categorical_features, target_feature)
    save_data(cleaned_data, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess a CSV or Excel file.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV or Excel file.")
    parser.add_argument('output_file', type=str, help="Path to save the output CSV file.")
    parser.add_argument('--remove_features', nargs='*', help="List of features to remove from the data.")
    parser.add_argument('--numerical_features', nargs='*', help="List of numerical features to process.")
    parser.add_argument('--categorical_features', nargs='*', help="List of categorical features to process.")
    parser.add_argument('--target_feature', type=str, default='target', help="The target feature for prediction.")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.remove_features, args.numerical_features, args.categorical_features, args.target_feature)

    # test code on dataset/depression.csv
    # python preprocess.py dataset/depression.csv dataset/depression_cleaned.csv --remove_features 'Response ID' --categorical_features 'Gender' --target_feature 'Depression Risk'