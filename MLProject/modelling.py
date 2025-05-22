import argparse
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def main(data_path):
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Model_Training_Default_Parameters")

    df = pd.read_csv(data_path)
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    X = df.drop(columns=['Depression'])
    y = df['Depression']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv("train_features.csv", index=False)
    y_train.to_csv("train_labels.csv", index=False)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=None),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
        "SVM": SVC(random_state=42, kernel='rbf', C=1.0, probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, algorithm='auto'),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "XGBoost": xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1, num_leaves=31, n_estimators=100),
        "Extra Trees": ExtraTreesClassifier(random_state=42, n_estimators=100, max_depth=None),
        "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=50, learning_rate=1.0),
        "Ridge Classifier": RidgeClassifier(alpha=1.0),
    }

    model_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, artifact_path=f"model_{name.replace(' ', '_')}", signature=signature)
        mlflow.log_params(model.get_params())

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric(f"{name}_accuracy", accuracy)
        mlflow.log_metric(f"{name}_f1_score", f1)
        mlflow.log_metric(f"{name}_precision", precision)
        mlflow.log_metric(f"{name}_recall", recall)

        class_report = classification_report(y_test, y_pred, output_dict=True)
        for label in class_report.keys():
            if label.isdigit():
                mlflow.log_metric(f"{name}_precision_label_{label}", class_report[label]['precision'])
                mlflow.log_metric(f"{name}_recall_label_{label}", class_report[label]['recall'])

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("conf_matrix.png")
        mlflow.log_artifact("conf_matrix.png")
        plt.close()

        mlflow.log_artifact("train_features.csv")
        mlflow.log_artifact("train_labels.csv")

        os.system("pip freeze > requirements.txt")
        mlflow.log_artifact("requirements.txt")

        model_results[name] = accuracy

    best_model_name = max(model_results, key=model_results.get)
    best_accuracy = model_results[best_model_name]
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    # Hyperparameter tuning grids
    param_grids = {
        "Logistic Regression": {
            'C': [0.1, 1, 10],
            'penalty': ['l2', 'none'],
            'solver': ['lbfgs']
        },
        "Decision Tree": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        "Gradient Boosting": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        "K-Nearest Neighbors": {
            'n_neighbors': [3, 5, 10],
            'algorithm': ['auto', 'ball_tree']
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0]
        },
        "LightGBM": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50],
            'max_depth': [3, 5]
        },
        "Extra Trees": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        "AdaBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0]
        },
        "Ridge Classifier": {
            'alpha': [0.1, 1.0, 10.0]
        }
    }

    if best_model_name in param_grids:
        mlflow.set_experiment(f"Tuning_{best_model_name.replace(' ', '_')}")
        best_model = models[best_model_name]
        param_grid = param_grids[best_model_name]

        # Jalankan tuning tanpa start_run, karena run sudah aktif via mlflow run
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model_tuned = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model_tuned.predict(X_test)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(best_model_tuned, "model", signature=signature)
        mlflow.log_params(best_params)

        acc_train = accuracy_score(y_train, best_model_tuned.predict(X_train))
        acc_test = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric("tuned_train_accuracy", acc_train)
        mlflow.log_metric("tuned_test_accuracy", acc_test)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        report = classification_report(y_test, y_pred, output_dict=True)
        for label in report:
            if label.isdigit():
                mlflow.log_metric(f"precision_label_{label}", report[label]['precision'])
                mlflow.log_metric(f"recall_label_{label}", report[label]['recall'])

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{best_model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("tuned_conf_matrix.png")
        mlflow.log_artifact("tuned_conf_matrix.png")
        plt.close()

        mlflow.log_artifact("train_features.csv")
        mlflow.log_artifact("train_labels.csv")
        os.system("pip freeze > requirements.txt")
        mlflow.log_artifact("requirements.txt")

        print(f"\n🔧 Tuned model: {best_model_name}")
        print(f"Best Params: {best_params}")
        print(f"Train Acc: {acc_train:.4f} | Test Acc: {acc_test:.4f} | F1: {f1:.4f}")

        # Register tuned model
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        client = MlflowClient()
        model_name = best_model_name.replace(" ", "_")

        try:
            client.create_registered_model(model_name)
        except Exception:
            print(f"Model '{model_name}' already registered, continuing...")

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        print(f"✅ Registered model version: {model_version.version} for model '{model_name}'")

        existing_version_info = client.get_model_version(name=model_name, version=model_version.version)
        existing_model_source = existing_version_info.source

        new_version = client.create_model_version(
            name=model_name,
            source=existing_model_source,
            run_id=None
        )

        print(f"🌀 Created duplicate model version: {new_version.version}")

        client.transition_model_version_stage(
            name=model_name,
            version=new_version.version,
            stage="Production"
        )

        print(f"🚀 Model version {new_version.version} is now in PRODUCTION stage ✅")

    else:
        print(f"No hyperparameter grid available for {best_model_name}, skipping tuning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='student-depression-dataset_preprocessing.csv')
    args = parser.parse_args()
    main(args.data_path)