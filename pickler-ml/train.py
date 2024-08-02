import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train models with specified hyperparameters.")
    parser.add_argument('input_file', type=str, help='Path to the input CSV or XLSX or XLS file.')
    parser.add_argument('target_feature', type=str, help='The target feature for classification.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split (0.0 to 1.0)')
    parser.add_argument('models_output', type=str, help='Folder to save the trained models.')
    parser.add_argument('--models', nargs='+', default=['SVM', 'DecisionTree', 'Bagging', 'AdaBoost', 'XGBoost', 'Stacking', 'RandomForest', 'NeuralNetwork', 'NaiveBayes'], help='List of models to train.')
    parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters.json', help='Path to JSON file containing hyperparameters for models.')
    args = parser.parse_args()
    
    return args 

def load_data(input_file, target_feature):
    data = pd.read_csv(input_file)
    X = data.drop(columns=[target_feature])
    y = data[target_feature]
    return X, y

def load_hyperparameters(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def train_models(X, y, models, hyperparameters, models_output, test_size):
    os.makedirs(models_output, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model_objects = {
        'SVM': SVC(**hyperparameters.get('SVM', {})),
        'DecisionTree': DecisionTreeClassifier(**hyperparameters.get('DecisionTree', {})),
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), **hyperparameters.get('Bagging', {})),
        'AdaBoost': AdaBoostClassifier(**hyperparameters.get('AdaBoost', {})),
        'XGBoost': XGBClassifier(**hyperparameters.get('XGBoost', {})),
        'Stacking': StackingClassifier(estimators=hyperparameters.get('Stacking', {}).get('estimators', [('rf', RandomForestClassifier()), ('svm', make_pipeline(StandardScaler(), SVC()))]),
                                       final_estimator=hyperparameters.get('Stacking', {}).get('final_estimator', LogisticRegression(max_iter=1000)),
                                       cv=hyperparameters.get('Stacking', {}).get('cv', 5)),
        'RandomForest': RandomForestClassifier(**hyperparameters.get('RandomForest', {})),
        'NeuralNetwork': MLPClassifier(**hyperparameters.get('NeuralNetwork', {})),
        'NaiveBayes': GaussianNB()
    }
    
    for model_name in models:
        model = model_objects.get(model_name)
        if model is not None:
            try:
                model.fit(X_train, y_train)
                train_predictions = model.predict(X_train)
                test_predictions = model.predict(X_test)

                train_accuracy = accuracy_score(y_train, train_predictions)
                test_accuracy = accuracy_score(y_test, test_predictions)

                print(f"{model_name} - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")
                
                model_path = os.path.join(models_output, f"{model_name}.pkl")
                with open(model_path, 'wb') as file:
                    pickle.dump(model, file)
                print(f"Saved {model_name} model to {model_path}.")
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
        else:
            print(f"Model {model_name} not found in model_objects dictionary")

def main():
    args = parse_arguments()
    
    hyperparameters = load_hyperparameters(args.hyperparameters_file)
    
    X, y = load_data(args.input_file, args.target_feature)
    train_models(X, y, args.models, hyperparameters, args.models_output, args.test_size)

if __name__ == '__main__':
    main()