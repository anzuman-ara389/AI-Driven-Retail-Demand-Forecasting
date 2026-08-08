import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.database import get_connection, init_db
from src.logging_utils import save_json
from src.preprocess import preprocess_data


MODEL_PATH = "models/demand_model.pkl"
FEATURES_PATH = "models/model_features.pkl"

METRICS_PATH = "artifacts/model_metrics.csv"
MODEL_COMPARISON_PATH = "artifacts/model_comparison.csv"
TRAINING_SUMMARY_PATH = "artifacts/training_summary.json"
FEATURE_IMPORTANCE_PATH = "artifacts/feature_importance.csv"


FEATURE_COLUMNS = [
    "product_encoded",
    "category_encoded",
    "store_id",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "promotion",
    "temperature",
    "unit_price",
    "expiry_days",
    "short_expiry",
    "promotion_active",
]

TARGET_COLUMN = "units_sold"

TEST_SIZE = 0.20
CV_SPLITS = 3
RANDOM_STATE = 42


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def prepare_data(df):

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target: {TARGET_COLUMN}"
        )

    data = df.copy()

    if "date" in data.columns:

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        data = data.sort_values(
            "date",
            kind="stable",
        ).reset_index(drop=True)

    X = data[FEATURE_COLUMNS].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.fillna(0)

    y = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    if y.notna().sum() == 0:
        raise ValueError(
            "Target column has no valid values."
        )

    y = y.fillna(
        y.median()
    )

    if len(X) < 20:
        raise ValueError(
            "Not enough rows for model training."
        )

    return X, y


def split_data(X, y):

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    if split_index <= 0 or split_index >= len(X):
        raise ValueError(
            "Invalid train-test split."
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):

    fitted_model = clone(model)

    fitted_model.fit(
        X_train,
        y_train,
    )

    train_predictions = fitted_model.predict(
        X_train
    )

    test_predictions = fitted_model.predict(
        X_test
    )

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    cv = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    cv_scores = cross_val_score(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    cv_rmse_scores = -cv_scores

    generalization_gap = (
        test_metrics["rmse"]
        - train_metrics["rmse"]
    )

    absolute_gap = abs(
        generalization_gap
    )

    return {
        "model": fitted_model,

        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],

        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],

        "cv_rmse_mean": float(
            cv_rmse_scores.mean()
        ),

        "cv_rmse_std": float(
            cv_rmse_scores.std()
        ),

        "generalization_gap": float(
            generalization_gap
        ),

        "absolute_gap": float(
            absolute_gap
        ),
    }


def get_candidate_models():

    models = {
        "RandomForestRegressor":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    }

    if XGBOOST_AVAILABLE:

        models["XGBoostRegressor"] = (
            XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.80,
                colsample_bytree=0.80,
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                n_jobs=-1,
            )
        )

    else:

        print(
            "XGBoost is not installed. "
            "Only Random Forest will be trained."
        )

    return models


def save_feature_importance(
    model,
    model_name,
):

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return None

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
            "model_name": model_name,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    return FEATURE_IMPORTANCE_PATH


def register_model(
    model_name,
    model_path,
    result,
    training_rows,
):

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_registry (
                model_name,
                model_path,
                mae,
                rmse,
                r2,
                training_rows,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                model_path,
                result["mae"],
                result["rmse"],
                result["r2"],
                int(training_rows),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()

    finally:

        conn.close()


def train_model():

    init_db()

    df = preprocess_data()

    X, y = prepare_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    candidate_models = get_candidate_models()

    if not candidate_models:
        raise RuntimeError(
            "No candidate model is available."
        )

    comparison_rows = []

    best_model_name = None
    best_result = None

    for model_name, model in candidate_models.items():

        print(
            f"\nTraining model: {model_name}"
        )

        result = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        row = {
            "model_name": model_name,

            "train_mae": round(
                result["train_mae"],
                4,
            ),

            "train_rmse": round(
                result["train_rmse"],
                4,
            ),

            "train_r2": round(
                result["train_r2"],
                4,
            ),

            "mae": round(
                result["mae"],
                4,
            ),

            "rmse": round(
                result["rmse"],
                4,
            ),

            "r2": round(
                result["r2"],
                4,
            ),

            "cv_rmse_mean": round(
                result["cv_rmse_mean"],
                4,
            ),

            "cv_rmse_std": round(
                result["cv_rmse_std"],
                4,
            ),

            "generalization_gap": round(
                result["generalization_gap"],
                4,
            ),

            "absolute_gap": round(
                result["absolute_gap"],
                4,
            ),

            "training_rows": int(
                len(X_train)
            ),

            "testing_rows": int(
                len(X_test)
            ),

            "cv_splits": CV_SPLITS,

            "created_at":
                datetime.now().isoformat(),
        }

        comparison_rows.append(
            row
        )

        print(
            "Train RMSE:",
            row["train_rmse"],
        )

        print(
            "Test RMSE:",
            row["rmse"],
        )

        print(
            "Test R2:",
            row["r2"],
        )

        print(
            "CV RMSE:",
            row["cv_rmse_mean"],
        )

        print(
            "Generalization gap:",
            row["generalization_gap"],
        )

        print(
            "Absolute gap:",
            row["absolute_gap"],
        )

        if (
            best_result is None
            or result["cv_rmse_mean"]
            < best_result["cv_rmse_mean"]
        ):

            best_result = result
            best_model_name = model_name

    if best_result is None:
        raise RuntimeError(
            "No model result was produced."
        )

    best_model = best_result["model"]

    os.makedirs(
        "models",
        exist_ok=True,
    )

    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    versioned_model_path = (
        f"models/demand_model_{timestamp}.pkl"
    )

    joblib.dump(
        best_model,
        versioned_model_path,
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURES_PATH,
    )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    feature_importance_path = (
        save_feature_importance(
            best_model,
            best_model_name,
        )
    )

    metrics = {
        "model_name": best_model_name,

        "mae": round(
            best_result["mae"],
            4,
        ),

        "rmse": round(
            best_result["rmse"],
            4,
        ),

        "r2": round(
            best_result["r2"],
            4,
        ),

        "train_mae": round(
            best_result["train_mae"],
            4,
        ),

        "train_rmse": round(
            best_result["train_rmse"],
            4,
        ),

        "train_r2": round(
            best_result["train_r2"],
            4,
        ),

        "cv_rmse_mean": round(
            best_result["cv_rmse_mean"],
            4,
        ),

        "cv_rmse_std": round(
            best_result["cv_rmse_std"],
            4,
        ),

        "generalization_gap": round(
            best_result["generalization_gap"],
            4,
        ),

        "absolute_gap": round(
            best_result["absolute_gap"],
            4,
        ),

        "training_rows": int(
            len(X_train)
        ),

        "testing_rows": int(
            len(X_test)
        ),

        "total_rows": int(
            len(df)
        ),

        "target_column": TARGET_COLUMN,

        "cv_method": "TimeSeriesSplit",

        "cv_splits": CV_SPLITS,

        "selection_metric": "cv_rmse_mean",

        "model_path": versioned_model_path,

        "latest_model_path": MODEL_PATH,

        "features_path": FEATURES_PATH,

        "feature_importance_path":
            feature_importance_path,

        "created_at":
            datetime.now().isoformat(),
    }

    pd.DataFrame(
        [metrics]
    ).to_csv(
        METRICS_PATH,
        index=False,
    )

    training_summary = {
        **metrics,

        "features_used":
            FEATURE_COLUMNS,

        "models_compared":
            list(candidate_models.keys()),

        "train_test_split":
            "chronological_80_20",
    }

    save_json(
        TRAINING_SUMMARY_PATH,
        training_summary,
    )

    register_model(
        best_model_name,
        versioned_model_path,
        best_result,
        len(X_train),
    )

    print(
        "\nModel training completed."
    )

    print(
        "Models compared:",
        list(candidate_models.keys()),
    )

    print(
        "Best model:",
        best_model_name,
    )

    print(
        "Selection metric:",
        "CV RMSE",
    )

    print(
        "Test MAE:",
        metrics["mae"],
    )

    print(
        "Test RMSE:",
        metrics["rmse"],
    )

    print(
        "Test R2:",
        metrics["r2"],
    )

    print(
        "Train RMSE:",
        metrics["train_rmse"],
    )

    print(
        "CV RMSE:",
        metrics["cv_rmse_mean"],
    )

    print(
        "CV standard deviation:",
        metrics["cv_rmse_std"],
    )

    print(
        "Generalization gap:",
        metrics["generalization_gap"],
    )

    print(
        "Absolute gap:",
        metrics["absolute_gap"],
    )

    print(
        "Versioned model:",
        versioned_model_path,
    )

    print(
        "Latest model:",
        MODEL_PATH,
    )

    print(
        "Model comparison:",
        MODEL_COMPARISON_PATH,
    )

    print(
        "Feature importance:",
        feature_importance_path,
    )

    return metrics


if __name__ == "__main__":
    train_model()