import argparse
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder
import dagshub


def main(data_path):
    # Initialize dagshub setup (token read automatically from env var DAGSHUB_USER_TOKEN)
    dagshub.init(
        repo_owner='Rendika7',
        repo_name='Eksperimen_SML_Rendika-nurhartanto-suharto',
        mlflow=True
    )

    # Set MLflow tracking URI and experiment name
    mlflow.set_tracking_uri("https://dagshub.com/Rendika7/Eksperimen_SML_Rendika-nurhartanto-suharto.mlflow")
    mlflow.set_experiment("Model_Training_Default_Parameters")

    # Load the dataset
    df = pd.read_csv(data_path)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoded_data = {}
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
        encoded_data[col] = list(label_encoder.classes_)

    # Prepare features and target
    X = df.drop(columns=['Depression'])
    y = df['Depression']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
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

    # Store results for model comparison
    model_results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log model artifact
            mlflow.sklearn.log_model(model, f"{name}_model")

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            # Log params (example for some models)
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

            # Log per-class precision and recall
            class_report = classification_report(y_test, y_pred, output_dict=True)
            for label in class_report.keys():
                if label.isdigit():
                    mlflow.log_metric(f"{name}_precision_label_{label}", class_report[label]['precision'])
                    mlflow.log_metric(f"{name}_recall_label_{label}", class_report[label]['recall'])

            model_results[name] = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }

    best_model_name = max(model_results, key=lambda k: model_results[k]["accuracy"])
    print(f"Best model for tuning: {best_model_name}")

    # Hyperparameter grids
    param_grids = {
        "Logistic Regression": {
            'C': [0.1, 1, 10],
            'penalty': ['l2', 'none']
        },
        "Decision Tree": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "Gradient Boosting": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 0.9, 1.0]
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        "K-Nearest Neighbors": {
            'n_neighbors': [3, 5, 10],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        "Naive Bayes": {},
        "XGBoost": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 0.9, 1.0]
        },
        "LightGBM": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50],
            'max_depth': [3, 5]
        },
        "Extra Trees": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        "AdaBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0, 1.5],
        },
        "Ridge Classifier": {
            'alpha': [0.1, 1.0, 10.0]
        }
    }

    # Hyperparameter tuning
    mlflow.set_experiment(f"Tuning_{best_model_name}")
    best_model_param_grid = param_grids.get(best_model_name, {})

    if best_model_param_grid:
        with mlflow.start_run(run_name=f"{best_model_name}_Tuning"):
            grid_search = GridSearchCV(estimator=models[best_model_name], param_grid=best_model_param_grid,
                                       cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            best_model_tuned = grid_search.best_estimator_

            mlflow.log_params(best_params)
            mlflow.log_metric("best_accuracy", grid_search.best_score_)

            y_train_pred = best_model_tuned.predict(X_train)
            y_test_pred = best_model_tuned.predict(X_test)

            tuned_train_accuracy = accuracy_score(y_train, y_train_pred)
            tuned_test_accuracy = accuracy_score(y_test, y_test_pred)
            tuned_f1 = f1_score(y_test, y_test_pred, average='weighted')
            tuned_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            tuned_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)

            mlflow.log_metric("tuned_train_accuracy", tuned_train_accuracy)
            mlflow.log_metric("tuned_test_accuracy", tuned_test_accuracy)
            mlflow.log_metric("tuned_f1", tuned_f1)
            mlflow.log_metric("tuned_precision", tuned_precision)
            mlflow.log_metric("tuned_recall", tuned_recall)

            # Log per-class precision and recall for tuned model
            class_report = classification_report(y_test, y_test_pred, output_dict=True)
            for label in class_report.keys():
                if label.isdigit():
                    mlflow.log_metric(f"{best_model_name}_precision_label_{label}", class_report[label]['precision'])
                    mlflow.log_metric(f"{best_model_name}_recall_label_{label}", class_report[label]['recall'])

            # Log the tuned model artifact
            mlflow.sklearn.log_model(best_model_tuned, f"{best_model_name}_tuned_model")

            # Register model to MLflow Model Registry
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{best_model_name}_tuned_model"
            client = MlflowClient()

            model_name = best_model_name.replace(" ", "_")

            # Try to create registered model, ignore if exists
            try:
                client.create_registered_model(model_name)
            except Exception:
                pass  # model already exists

            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            )
            print(f"Registered model version: {model_version.version} for model '{model_name}'")

            # Transition the model version to Production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            print(f"Model version {model_version.version} is now in Production stage")

            # Print tuned metrics
            print(f"Tuned Train Accuracy: {tuned_train_accuracy}")
            print(f"Tuned Test Accuracy: {tuned_test_accuracy}")
            print(f"Tuned F1-Score: {tuned_f1}")
            print(f"Tuned Precision: {tuned_precision}")
            print(f"Tuned Recall: {tuned_recall}")
    else:
        print(f"No hyperparameter tuning available for {best_model_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='student-depression-dataset_preprocessing.csv')
    args = parser.parse_args()
    main(args.data_path)