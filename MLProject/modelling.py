import argparse
import pandas as pd
import os
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb  # type: ignore
import lightgbm as lgb  # type: ignore
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import dagshub

def main(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # List of categorical columns to encode
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Apply Label Encoding to each categorical column and store the encoding
    encoded_data = {}

    for col in categorical_columns:
        # Fit the encoder and transform the column
        df[col] = label_encoder.fit_transform(df[col])
        encoded_data[col] = list(label_encoder.classes_)  # Store original categories for reference

    # Then you can proceed with splitting features and target for model training
    X = df.drop(columns=['Depression'])
    y = df['Depression']

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define all models with their default parameters
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=None),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
        "SVM": SVC(random_state=42, kernel='rbf', C=1.0),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, algorithm='auto'),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "XGBoost": xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1, num_leaves=31, n_estimators=100),
        "Extra Trees": ExtraTreesClassifier(random_state=42, n_estimators=100, max_depth=None),
        "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=50, learning_rate=1.0),
        "Ridge Classifier": RidgeClassifier(alpha=1.0),
    }

    # Initialize dagshub setup
    dagshub.init(repo_owner='Rendika7', repo_name='Eksperimen_SML_Rendika-nurhartanto-suharto', mlflow=True)

    # Set MLflow tracking URI (remote DagsHub)
    mlflow.set_tracking_uri("https://dagshub.com/Rendika7/Eksperimen_SML_Rendika-nurhartanto-suharto.mlflow")
    mlflow.set_experiment("Model_Training_Default_Parameters")

    # Initialize a dictionary to store the results
    model_results = {}

    for name, model in models.items():
        # Start a new run for each model
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log the model as an artifact
            mlflow.sklearn.log_model(model, f"{name}_model")

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            # Log parameters (all parameters relevant to the model)
            if name == "Logistic Regression":
                mlflow.log_param(f"{name}_C", model.C)
                mlflow.log_param(f"{name}_penalty", model.penalty)
            elif name == "Decision Tree":
                mlflow.log_param(f"{name}_max_depth", model.max_depth)
            elif name == "Random Forest":
                mlflow.log_param(f"{name}_n_estimators", model.n_estimators)
                mlflow.log_param(f"{name}_max_depth", model.max_depth)
            elif name == "Gradient Boosting":
                mlflow.log_param(f"{name}_n_estimators", model.n_estimators)
                mlflow.log_param(f"{name}_learning_rate", model.learning_rate)
            elif name == "SVM":
                mlflow.log_param(f"{name}_kernel", model.kernel)
                mlflow.log_param(f"{name}_C", model.C)
            elif name == "K-Nearest Neighbors":
                mlflow.log_param(f"{name}_n_neighbors", model.n_neighbors)
                mlflow.log_param(f"{name}_algorithm", model.algorithm)
            elif name == "Naive Bayes":
                mlflow.log_param(f"{name}_var_smoothing", model.var_smoothing)
            elif name == "XGBoost":
                mlflow.log_param(f"{name}_n_estimators", model.n_estimators)
                mlflow.log_param(f"{name}_max_depth", model.max_depth)
            elif name == "LightGBM":
                mlflow.log_param(f"{name}_num_leaves", model.num_leaves)
                mlflow.log_param(f"{name}_n_estimators", model.n_estimators)
            elif name == "Extra Trees":
                mlflow.log_param(f"{name}_n_estimators", model.n_estimators)
                mlflow.log_param(f"{name}_max_depth", model.max_depth)
            elif name == "AdaBoost":
                mlflow.log_param(f"{name}_n_estimators", model.n_estimators)
                mlflow.log_param(f"{name}_learning_rate", model.learning_rate)
            elif name == "Ridge Classifier":
                mlflow.log_param(f"{name}_alpha", model.alpha)

            # Log metrics
            mlflow.log_metric(f"{name}_accuracy", accuracy)
            mlflow.log_metric(f"{name}_f1", f1)
            mlflow.log_metric(f"{name}_precision", precision)
            mlflow.log_metric(f"{name}_recall", recall)

            # Classification Report to get precision and recall for each label
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Log precision and recall for each label manually
            for label in class_report.keys():
                if label.isdigit():  # Only log metrics for labels (not 'accuracy', 'macro avg', etc.)
                    mlflow.log_metric(f"{name}_precision_label_{label}", class_report[label]['precision'])
                    mlflow.log_metric(f"{name}_recall_label_{label}", class_report[label]['recall'])

            # Store results for model comparison
            model_results[name] = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }

    # Find the best model based on test accuracy
    best_model_name = max(model_results, key=lambda k: model_results[k]["accuracy"])
    print(f"Best model for tuning: {best_model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='student-depression-dataset_preprocessing.csv')
    args = parser.parse_args()
    main(args.data_path)
