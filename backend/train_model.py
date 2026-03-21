import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(num_samples=5000, seed=42):
    np.random.seed(seed)

    data = []

    # LOW RISK (0)
    for _ in range(num_samples // 3):
        wbc = np.clip(np.random.normal(7500, 1500), 4500, 11000)
        rbc = np.clip(np.random.normal(4.8, 0.4), 4.0, 5.5)
        hb = np.clip(np.random.normal(14.5, 1.0), 12, 17)
        platelets = np.clip(np.random.normal(250000, 40000), 150000, 400000)
        data.append([wbc, rbc, hb, platelets, 0])

    # MEDIUM RISK (1)
    for _ in range(num_samples // 3):
        wbc = np.clip(np.random.normal(13000, 4000), 9000, 25000)
        rbc = np.clip(np.random.normal(4.0, 0.6), 3.0, 5.0)
        hb = np.clip(np.random.normal(10.5, 2), 8, 12)
        platelets = np.clip(np.random.normal(140000, 50000), 80000, 200000)
        data.append([wbc, rbc, hb, platelets, 1])

    # HIGH RISK (2)
    for _ in range(num_samples // 3):
        wbc = np.clip(np.random.normal(50000, 30000), 20000, 150000)
        rbc = np.clip(np.random.normal(3.0, 1.0), 1.5, 4.5)
        hb = np.clip(np.random.normal(7, 2), 4, 10)
        platelets = np.clip(np.random.normal(80000, 40000), 20000, 150000)
        data.append([wbc, rbc, hb, platelets, 2])

    return pd.DataFrame(data, columns=["WBC", "RBC", "Hb", "Platelets", "Label"])
def load_data(data_path: Path) -> pd.DataFrame:
    """Load training data from disk or generate synthetic examples."""
    if data_path.exists():
        df = pd.read_csv(data_path)
        if "Label" in df.columns and not df.empty:
            print(f"Loaded {len(df)} records from {data_path}")

            # If we only have a small real dataset, augment it with synthetic records
            # so that cross-validation and hyperparameter search behave more reliably.
            if len(df) < 200:
                augment_size = 200 - len(df)
                print(f"Augmenting with {augment_size} synthetic records to improve training stability.")
                df = pd.concat([df, generate_synthetic_data(num_samples=augment_size)], ignore_index=True)

            return df
        print(f"Data file {data_path} existed but did not contain valid labels. Falling back to synthetic data.")

    print("No valid CSV found; generating synthetic training data.")
    return generate_synthetic_data(num_samples=5000)


def train_and_save_model():
    workspace_root = Path(__file__).resolve().parent
    data_path = workspace_root / "data" / "cbc_data.csv"

    df = load_data(data_path)
    X = df[["WBC", "RBC", "Hb", "Platelets"]]
    y = df["Label"]

    # For small datasets, ensure the test set has at least one sample per class.
    n_samples = len(y)
    n_classes = y.nunique()
    desired_test = max(n_classes, int(n_samples * 0.2))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=desired_test, stratify=y, random_state=42
        )
    except ValueError:
        # Fallback: use a minimal stratified split when data is very small
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_classes, stratify=y, random_state=42
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")),
    ])

    param_dist = {
        "classifier__n_estimators": [100, 250, 500, 750, 1000],
        "classifier__max_depth": [None, 10, 20, 30, 50],
        "classifier__min_samples_split": [2, 3, 5, 8],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", None],
    }

    min_class_count = y_train.value_counts().min()
    use_search = min_class_count >= 2 and len(y_train) >= 10

    if use_search:
        # Ensure the number of CV splits is not greater than the minimum class count.
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=40,
            scoring="roc_auc_ovr",
            n_jobs=-1,
            cv=cv,
            random_state=42,
            verbose=1,
            refit=True,
        )

        search.fit(X_train, y_train)

        best = search.best_estimator_
        print("\nBest hyperparameters:")
        print(search.best_params_)

        calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=cv)
        calibrated.fit(X_train, y_train)
    else:
        # When data is too small for reliable cross-validation, train a single model.
        print("Training on full dataset without hyperparameter search due to small sample size.")
        best = pipeline
        best.fit(X_train, y_train)

        # Skip probability calibration in very-small-data situations.
        calibrated = best

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    print("\nTest set evaluation:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_dir = workspace_root / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "leukemia_model.pkl"
    joblib.dump(calibrated, model_path)
    print(f"\nCalibrated model trained and saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
