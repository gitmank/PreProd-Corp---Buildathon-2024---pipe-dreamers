import pandas as pd
import numpy as np
import os

def clean_data(file_path, selected_features, features_to_convert):
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")
    
    print("Data loaded successfully.")
    
    # Show initial data dimensions and preview
    rows, cols = df.shape
    print(f"\nInitial Data Dimensions: {rows} rows, {cols} columns")
    print("\nDataset Preview (first 6 rows):")
    print(df.head(6))
    
    # Select features
    df = df[selected_features]
    print("\nSelected features:", ', '.join(selected_features))
    
    # Remove null rows
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    print(f"\nRemoved {rows_removed} rows with null values.")
    
    # Convert to numeric
    for feature in features_to_convert:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    print(f"\nConverted the following features to numeric: {', '.join(features_to_convert)}")
    
    # Final data dimensions
    rows, cols = df.shape
    print(f"\nFinal Data Dimensions: {rows} rows, {cols} columns")
    
    return df

def save_clean_data(file_path, selected_features, features_to_convert, output_path):
    try:
        cleaned_df = clean_data(file_path, selected_features, features_to_convert)
        
        # Save the cleaned dataset
        cleaned_df.to_csv(output_path, index=False)
        print(f"\nCleaned dataset saved to {output_path}")
        
        return True, "Data cleaned and saved successfully."
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

# Example usage:
if __name__ == "__main__":
    # These would be passed from the frontend
    file_path = "depressionurture.csv"
    selected_features = ["feature1", "feature2", "feature3"]
    features_to_convert = ["feature1", "feature3"]
    output_path = "path/to/your/output/cleaned_dataset.csv"
    
    success, message = save_clean_data(file_path, selected_features, features_to_convert, output_path)
    print(message)