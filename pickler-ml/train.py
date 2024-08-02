import pandas as pd
import numpy as np
import pickle
import os
import warnings
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import logging

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    'SVM': make_pipeline(StandardScaler(), SVC(probability=True)),
    'DecisionTree': DecisionTreeClassifier(),
    'Bagging': BaggingClassifier(),
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('svm', make_pipeline(StandardScaler(), SVC(probability=True)))
        ],
        final_estimator=LogisticRegression(max_iter=1000)
    )
}

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV file: {file_path}")
        raise

def split_data(data, target_feature, test_size=0.2, random_state=42):
    if target_feature not in data.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in the dataset")

    X = data.drop(columns=[target_feature])
    y = data[target_feature]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_neural_network(input_dim, num_classes):
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    if isinstance(model, tf.keras.Model):
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # For binary classification
    if y_pred_proba.shape[1] == 2:
        auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc
    }

def hyperparameter_tuning(model, X_train, y_train):
    param_distributions = {
        'SVM': {'svc__C': [0.1, 1, 10], 'svc__kernel': ['rbf', 'linear']},
        'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
        'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.1, 0.3]}
    }

    model_name = type(model).__name__
    if model_name in param_distributions:
        tuned_model = RandomizedSearchCV(model, param_distributions[model_name], n_iter=10, cv=3, n_jobs=-1)
        tuned_model.fit(X_train, y_train)
        return tuned_model.best_estimator_
    return model

def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test, output_folder):
    logger.info(f"Training {name}...")
    model = hyperparameter_tuning(model, X_train, y_train)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model with versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{name}_{timestamp}.pkl"
    with open(os.path.join(output_folder, model_filename), 'wb') as f:
        pickle.dump(model, f)
    
    return name, metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models_to_train, output_folder):
    results = {}
    
    for name, model in AVAILABLE_MODELS.items():
        if name in models_to_train:
            name, metrics = train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test, output_folder)
            results[name] = metrics

    if 'NeuralNetwork' in models_to_train:
        logger.info("Training Neural Network...")
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        nn_model = build_neural_network(input_dim, num_classes)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nn_model.save(os.path.join(output_folder, f'NeuralNetwork_{timestamp}.keras'))
        
        results['NeuralNetwork'] = evaluate_model(nn_model, X_test, y_test)

    return results

def feature_importance_analysis(X, y):
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    return feature_scores

def main(input_file, output_folder, target_feature, models_to_train, config_file=None):
    # Default configuration
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'feature_selection': {
            'enabled': False,
            'k_best': 10
        },
        'hyperparameter_tuning': {
            'enabled': True,
            'n_iter': 10,
            'cv': 3
        },
        'neural_network': {
            'epochs': 50,
            'batch_size': 32,
            'early_stopping_patience': 5
        }
    }

    # Load configuration if provided
    if config_file:
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)

    data = load_data(input_file)
    X_train, X_test, y_train, y_test = split_data(data, target_feature, 
                                                  test_size=config['test_size'], 
                                                  random_state=config['random_state'])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Perform feature importance analysis
    feature_scores = feature_importance_analysis(X_train, y_train)
    logger.info("Top 10 important features:")
    logger.info(feature_scores.head(10))

    # Apply feature selection if enabled
    if config['feature_selection']['enabled']:
        k_best = config['feature_selection']['k_best']
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        selected_features = X_train.columns[selector.get_support()]
        logger.info(f"Selected top {k_best} features: {', '.join(selected_features)}")

    # Update hyperparameter tuning settings
    if config['hyperparameter_tuning']['enabled']:
        global hyperparameter_tuning
        def hyperparameter_tuning(model, X_train, y_train):
            param_distributions = {
                'SVM': {'svc__C': [0.1, 1, 10], 'svc__kernel': ['rbf', 'linear']},
                'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
                'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
                'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.1, 0.3]}
            }

            model_name = type(model).__name__
            if model_name in param_distributions:
                tuned_model = RandomizedSearchCV(model, param_distributions[model_name], 
                                                 n_iter=config['hyperparameter_tuning']['n_iter'], 
                                                 cv=config['hyperparameter_tuning']['cv'], 
                                                 n_jobs=-1)
                tuned_model.fit(X_train, y_train)
                return tuned_model.best_estimator_
            return model

    # Update neural network settings
    global build_neural_network
    def build_neural_network(input_dim, num_classes):
        model = Sequential([
            tf.keras.Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, models_to_train, output_folder)

    # Save results
    with open(os.path.join(output_folder, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    for model_name, metrics in results.items():
        logger.info(f"{model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric} = {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate various ML models.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file.")
    parser.add_argument('output_dir', type=str, help="Folder to save the trained models and results.")
    parser.add_argument('--target_feature', type=str, required=True, help="The target feature for prediction.")
    parser.add_argument('--models', type=str, nargs='+', choices=list(AVAILABLE_MODELS.keys()) + ['NeuralNetwork'], 
                        default=list(AVAILABLE_MODELS.keys()) + ['NeuralNetwork'], 
                        help="List of models to train. If not provided, all models will be trained.")
    parser.add_argument('--config', type=str, help="Path to a JSON configuration file.")

    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.target_feature, args.models, args.config)