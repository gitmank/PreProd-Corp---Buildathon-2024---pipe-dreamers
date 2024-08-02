import pandas as pd
import numpy as np
import os
import json
import pyarrow.parquet as pq
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.covariance import EllipticEnvelope

def detect_file_type(file_path):
    """Detect the file type based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.csv']:
        return 'csv'
    elif ext in ['.xlsx', '.xls']:
        return 'excel'
    elif ext in ['.json']:
        return 'json'
    elif ext in ['.parquet']:
        return 'parquet'
    else:
        raise ValueError("Unsupported file format. Please use CSV, Excel, JSON, or Parquet files.")

def load_data(file_path, chunksize=None):
    """Load data from various file formats, with optional chunking for large files."""
    file_type = detect_file_type(file_path)
    
    if file_type == 'csv':
        return pd.read_csv(file_path, chunksize=chunksize)
    elif file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path, lines=True, chunksize=chunksize)
    elif file_type == 'parquet':
        return pq.read_table(file_path).to_pandas()

def detect_and_convert_dtypes(df):
    """Detect and convert data types, including datetime fields."""
    for col in df.columns:
        # Attempt to convert to datetime
        try:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors='ignore')
            continue
        except Exception as e:
            pass
        
        # Attempt to convert to numeric
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                pass
    
    return df

def handle_outliers(df, method='iqr', threshold=1.5):
    """Detect and handle outliers using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            df[col] = df[col].clip(lower_bound, upper_bound)
    elif method == 'zscore':
        outlier_detector = EllipticEnvelope(contamination=0.1)
        for col in numeric_cols:
            mask = outlier_detector.fit_predict(df[[col]])
            df.loc[mask == -1, col] = np.nan
    
    return df

def impute_missing_values(df, numeric_strategy='knn', categorical_strategy='most_frequent'):
    """Impute missing values using specified strategies."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) == 0 and len(categorical_cols) == 0:
        print("No columns available for imputation.")
        return df
    
    if numeric_strategy == 'knn':
        numeric_imputer = KNNImputer(n_neighbors=5)
    else:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
    
    categorical_imputer = SimpleImputer(strategy=categorical_strategy)
    
    if len(numeric_cols) > 0:
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    if len(categorical_cols) > 0:
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    return df

def normalize_data(df, method='standard'):
    """Normalize numeric data using specified method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns available for normalization.")
        return df
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def clean_data(file_path, selected_features, features_to_convert, encoding_type='onehot', target_feature=None, 
               outlier_method='iqr', impute_strategy='knn', normalize_method='standard', chunksize=None):
    """Main function to clean and preprocess data."""
    
    # Load data
    df = load_data(file_path, chunksize)
    if chunksize:
        df = pd.concat(df)
    print("Data loaded successfully.")
    
    # Show initial data dimensions and preview
    rows, cols = df.shape
    print(f"\nInitial Data Dimensions: {rows} rows, {cols} columns")
    print("\nDataset Preview (first 6 rows):")
    print(df.head(6))
    
    # Select features
    df = df[selected_features]
    print("\nSelected features:", ', '.join(selected_features))
    
    # Separate the target feature
    if target_feature and target_feature in df.columns:
        y = df[target_feature]
        df = df.drop(columns=[target_feature])
    else:
        y = None
    
    # Detect and convert data types
    df = detect_and_convert_dtypes(df)
    print("\nDetected and converted data types.")
    
    # Handle outliers
    df = handle_outliers(df, method=outlier_method)
    print(f"\nHandled outliers using {outlier_method} method.")
    
    # Check state of data before imputation
    print("\nData preview before imputation:")
    print(df.head())

    # Impute missing values
    df = impute_missing_values(df, numeric_strategy=impute_strategy)
    print(f"\nImputed missing values using {impute_strategy} strategy for numeric data.")
    
    # Check state of data before normalization
    print("\nData preview before normalization:")
    print(df.head())

    # Normalize data
    df = normalize_data(df, method=normalize_method)
    print(f"\nNormalized numeric data using {normalize_method} method.")
    
    # Convert features to numeric with specified encoding type
    for feature in features_to_convert:
        if feature in df.columns:
            if encoding_type == 'onehot':
                # Apply one-hot encoding
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(df[[feature]])
                encoded_df = pd.DataFrame(encoded_data, columns=[f"{feature}_{category}" for category in encoder.categories_[0]])
                df = pd.concat([df.drop(columns=[feature]), encoded_df], axis=1)
                print(f"\nApplied one-hot encoding to feature: {feature}")
            elif encoding_type == 'label':
                # Apply label encoding
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature])
                print(f"\nApplied label encoding to feature: {feature}")
            else:
                raise ValueError("Invalid encoding type. Please use 'onehot' or 'label'.")
    
    print(f"\nConverted the following features to numeric using {encoding_type} encoding: {', '.join(features_to_convert)}")
    
    # Add target feature back if it exists
    if y is not None:
        df[target_feature] = y.values
    
    # Final data dimensions
    rows, cols = df.shape
    print(f"\nFinal Data Dimensions: {rows} rows, {cols} columns")
    
    return df

def save_clean_data(file_path, selected_features, features_to_convert, output_dir, encoding_type='onehot', 
                    target_feature=None, outlier_method='iqr', impute_strategy='knn', normalize_method='standard', chunksize=None):
    """Clean data and save to output file with automatic naming."""
    try:
        # Generate output filename
        input_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{input_filename}_cleaned.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Clean data
        cleaned_df = clean_data(file_path, selected_features, features_to_convert, encoding_type, target_feature, 
                                outlier_method, impute_strategy, normalize_method, chunksize)
        
        # Save the cleaned dataset
        cleaned_df.to_csv(output_path, index=False)
        print(f"\nCleaned dataset saved to {output_path}")
        
        return True, "Data cleaned and saved successfully."
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

# Testing:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean and preprocess data from various file formats.")
    parser.add_argument('input_file', type=str, help="Path to the input file (CSV, Excel, JSON, or Parquet).")
    parser.add_argument('output_dir', type=str, help="Directory to save the output CSV file.")
    parser.add_argument('--selected_features', nargs='+', help="List of features to select from the data.")
    parser.add_argument('--features_to_convert', nargs='+', help="List of features to convert to numeric values.")
    parser.add_argument('--encoding_type', type=str, default='onehot', help="Encoding type: 'onehot' or 'label'.")
    parser.add_argument('--target_feature', type=str, help="The target feature for prediction.")
    
    # Default values for optional parameters
    outlier_method = 'iqr'
    impute_strategy = 'mean'
    normalize_method = 'standard'
    chunksize = None

    args = parser.parse_args()

    success, message = save_clean_data(
        args.input_file, 
        args.selected_features, 
        args.features_to_convert, 
        args.output_dir, 
        args.encoding_type, 
        args.target_feature, 
        outlier_method, 
        impute_strategy, 
        normalize_method, 
        chunksize
    )

    if success:
        print("Data processing completed successfully.")
    else:
        print(f"Error in data processing: {message}")