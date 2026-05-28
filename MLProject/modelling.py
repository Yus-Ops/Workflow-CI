"""
modelling.py - MLflow Project (Kriteria 3)
Training XGBoost untuk dengue, dijalankan via `mlflow run`.
"""
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main():
    data_path = os.path.join('dengue_preprocessing', 'dengue_clean.csv')
    df = pd.read_csv(data_path)
    tracking_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
    mlflow.set_tracking_uri("file:" + tracking_dir)
    X = df.drop('Result', axis=1)
    y = df['Result']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    # autolog untuk kemudahan CI (Basic-friendly)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42, eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    main()
