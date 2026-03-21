import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split


RAW_FEATURES = ["WBC", "RBC", "Hb", "Platelets"]
ENGINEERED_FEATURES = [
    "WBC_log",
    "Platelets_log",
    "Hb_RBC_ratio",
    "WBC_Platelets_ratio",
    "Cytopenia_count",
    "Leukocytosis_score",
    "Anemia_score",
    "Thrombocytopenia_score",
]
FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED_FEATURES
CLASS_LABELS = [0, 1, 2]

CLASS_PROFILES = {
    0: {
        "WBC": (7500, 1200, 4500, 11000),
        "RBC": (4.8, 0.35, 4.1, 5.8),
        "Hb": (14.2, 0.9, 12.0, 17.5),
        "Platelets": (260000, 45000, 150000, 420000),
    },
    1: {
        "WBC": (16000, 5000, 9000, 35000),
        "RBC": (3.9, 0.45, 2.8, 5.0),
        "Hb": (10.2, 1.4, 7.5, 12.2),
        "Platelets": (120000, 38000, 60000, 220000),
    },
    2: {
        "WBC": (65000, 28000, 20000, 200000),
        "RBC": (2.9, 0.55, 1.5, 4.2),
        "Hb": (7.1, 1.4, 3.5, 10.2),
        "Platelets": (70000, 26000, 10000, 150000),
    },
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    for col in RAW_FEATURES:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    features["WBC_log"] = np.log1p(features["WBC"].clip(lower=0))
    features["Platelets_log"] = np.log1p(features["Platelets"].clip(lower=0))
    features["Hb_RBC_ratio"] = features["Hb"] / features["RBC"].clip(lower=0.1)
    features["WBC_Platelets_ratio"] = features["WBC"] / features["Platelets"].clip(lower=1.0)
    features["Cytopenia_count"] = (
        (features["RBC"] < 4.0).astype(int)
        + (features["Hb"] < 12.0).astype(int)
        + (features["Platelets"] < 150000).astype(int)
    )
    features["Leukocytosis_score"] = (
        (features["WBC"] > 11000).astype(int)
        + (features["WBC"] > 20000).astype(int)
        + (features["WBC"] > 50000).astype(int)
    )
    features["Anemia_score"] = (
        (features["Hb"] < 12.0).astype(int)
        + (features["Hb"] < 10.0).astype(int)
        + (features["Hb"] < 8.0).astype(int)
    )
    features["Thrombocytopenia_score"] = (
        (features["Platelets"] < 150000).astype(int)
        + (features["Platelets"] < 100000).astype(int)
        + (features["Platelets"] < 50000).astype(int)
    )
    return features


def clean_labeled_data(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [col.strip() for col in frame.columns]
    required = RAW_FEATURES + ["Label"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    for col in RAW_FEATURES + ["Label"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=required)
    frame["Label"] = frame["Label"].astype(int)
    frame = frame[frame["Label"].isin(CLASS_LABELS)]
    return frame.reset_index(drop=True)


def sample_from_profile(rng: np.random.Generator, label: int, count: int) -> pd.DataFrame:
    profile = CLASS_PROFILES[label]
    rows = {}
    for feature in RAW_FEATURES:
        mean, std, low, high = profile[feature]
        rows[feature] = np.clip(rng.normal(mean, std, count), low, high)
    rows["Label"] = np.full(count, label, dtype=int)
    return pd.DataFrame(rows)


def augment_class_data(class_df: pd.DataFrame, label: int, target_count: int, seed: int) -> pd.DataFrame:
    if len(class_df) >= target_count:
        return class_df.copy()

    rng = np.random.default_rng(seed + label)
    needed = target_count - len(class_df)
    synthetic = sample_from_profile(rng, label, needed)

    if not class_df.empty:
        observed = class_df[RAW_FEATURES].sample(
            n=needed,
            replace=True,
            random_state=seed + label,
        ).reset_index(drop=True)
        jitter = pd.DataFrame(index=observed.index)
        for feature in RAW_FEATURES:
            _, std, low, high = CLASS_PROFILES[label][feature]
            feature_std = max(float(class_df[feature].std(ddof=0) or 0.0), std * 0.15)
            jitter[feature] = np.clip(
                observed[feature] + rng.normal(0, feature_std * 0.25, len(observed)),
                low,
                high,
            )
        jitter["Label"] = label
        synthetic = pd.concat([synthetic, jitter], ignore_index=True)
        synthetic = synthetic.sample(n=needed, random_state=seed + label).reset_index(drop=True)

    return pd.concat([class_df, synthetic], ignore_index=True)


def load_data(data_path: Path, seed: int = 42) -> pd.DataFrame:
    if data_path.exists():
        raw_df = pd.read_csv(data_path)
        df = clean_labeled_data(raw_df)
        if not df.empty:
            print(f"Loaded {len(df)} labeled records from {data_path}")
        else:
            print(f"Data file {data_path} had no usable labeled rows. Falling back to synthetic data.")
            df = pd.DataFrame(columns=RAW_FEATURES + ["Label"])
    else:
        print("No CSV found; training from synthetic baseline data.")
        df = pd.DataFrame(columns=RAW_FEATURES + ["Label"])

    # Ensure each class has enough support for 3-way risk prediction.
    augmented = []
    target_per_class = max(120, len(df))
    for label in CLASS_LABELS:
        class_df = df[df["Label"] == label].copy()
        augmented.append(augment_class_data(class_df, label, target_per_class, seed))

    result = pd.concat(augmented, ignore_index=True)
    print("Class balance after augmentation:")
    print(result["Label"].value_counts().sort_index().to_string())
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_and_save_model():
    seed = 42
    workspace_root = Path(__file__).resolve().parent
    data_path = workspace_root / "data" / "cbc_data.csv"

    df = load_data(data_path, seed=seed)
    features = engineer_features(df[RAW_FEATURES])
    X = features[FEATURE_COLUMNS]
    y = df["Label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    classifier = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    param_dist = {
        "n_estimators": [250, 400, 600, 800],
        "max_depth": [8, 12, 18, 24, None],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", "log2", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        classifier,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc_ovr",
        n_jobs=-1,
        cv=cv,
        random_state=seed,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=cv)
    calibrated.fit(X_train, y_train)

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print("\nBest hyperparameters:")
    print(search.best_params_)
    print("\nTest set evaluation:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    bundle = {
        "model": calibrated,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": CLASS_LABELS,
        "raw_features": RAW_FEATURES,
    }

    model_dir = workspace_root / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "leukemia_model.pkl"
    joblib.dump(bundle, model_path)
    print(f"\nRisk model bundle saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
