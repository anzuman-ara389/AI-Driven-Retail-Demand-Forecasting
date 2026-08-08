from datetime import datetime

import numpy as np
import pandas as pd

from src.database import get_connection, init_db
from src.logging_utils import append_csv_log


DRIFT_FEATURES = [
    "units_sold",
    "current_stock",
    "temperature",
    "waste_quantity",
]

DRIFT_THRESHOLD = 0.15
REFERENCE_WINDOW_SIZE = 1000
CURRENT_WINDOW_SIZE = 300
DRIFT_SUMMARY_PATH = "artifacts/drift_summary.csv"


def calculate_drift_score(reference_mean, current_mean):

    if abs(reference_mean) < 1e-9:
        return 0.0

    return abs(
        current_mean - reference_mean
    ) / abs(reference_mean)


def run_drift_check():

    init_db()

    conn = get_connection()

    try:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM raw_food_sales
            ORDER BY id ASC
            """,
            conn,
        )

        required_rows = (
            REFERENCE_WINDOW_SIZE
            + CURRENT_WINDOW_SIZE
        )

        if len(df) < required_rows:

            return {
                "message": (
                    "Not enough data for drift detection. "
                    f"Need at least {required_rows} rows."
                ),
                "available_rows": int(len(df)),
                "required_rows": required_rows,
                "drift_detected": False,
            }

        reference_df = df.iloc[
            -required_rows:-CURRENT_WINDOW_SIZE
        ].copy()

        current_df = df.iloc[
            -CURRENT_WINDOW_SIZE:
        ].copy()

        cursor = conn.cursor()

        drift_results = []
        drifted_features = []

        for feature in DRIFT_FEATURES:

            if feature not in df.columns:

                result = {
                    "feature_name": feature,
                    "reference_mean": None,
                    "current_mean": None,
                    "drift_score": None,
                    "drift_detected": False,
                    "status": "missing_feature",
                    "threshold": DRIFT_THRESHOLD,
                    "reference_window_size":
                        REFERENCE_WINDOW_SIZE,
                    "current_window_size":
                        CURRENT_WINDOW_SIZE,
                    "created_at":
                        datetime.now().isoformat(),
                }

                drift_results.append(result)

                continue

            reference_values = pd.to_numeric(
                reference_df[feature],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            current_values = pd.to_numeric(
                current_df[feature],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            reference_mean = float(
                np.nan_to_num(
                    reference_values.mean()
                )
            )

            current_mean = float(
                np.nan_to_num(
                    current_values.mean()
                )
            )

            drift_score = calculate_drift_score(
                reference_mean,
                current_mean,
            )

            feature_drift_detected = (
                drift_score > DRIFT_THRESHOLD
            )

            if feature_drift_detected:
                drifted_features.append(feature)

            result = {
                "feature_name": feature,
                "reference_mean": round(
                    reference_mean,
                    4,
                ),
                "current_mean": round(
                    current_mean,
                    4,
                ),
                "drift_score": round(
                    drift_score,
                    4,
                ),
                "drift_detected":
                    feature_drift_detected,
                "status": (
                    "drift"
                    if feature_drift_detected
                    else "stable"
                ),
                "threshold": DRIFT_THRESHOLD,
                "reference_window_size":
                    REFERENCE_WINDOW_SIZE,
                "current_window_size":
                    CURRENT_WINDOW_SIZE,
                "created_at":
                    datetime.now().isoformat(),
            }

            drift_results.append(result)

            cursor.execute(
                """
                INSERT INTO drift_reports (
                    feature_name,
                    reference_mean,
                    current_mean,
                    drift_score,
                    drift_detected,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feature,
                    reference_mean,
                    current_mean,
                    drift_score,
                    int(feature_drift_detected),
                    datetime.now().isoformat(),
                ),
            )

            append_csv_log(
                DRIFT_SUMMARY_PATH,
                result,
            )

        conn.commit()

        return {
            "drift_detected":
                len(drifted_features) > 0,
            "drifted_features":
                drifted_features,
            "drifted_feature_count":
                len(drifted_features),
            "threshold":
                DRIFT_THRESHOLD,
            "reference_window_size":
                REFERENCE_WINDOW_SIZE,
            "current_window_size":
                CURRENT_WINDOW_SIZE,
            "reference_rows":
                len(reference_df),
            "current_rows":
                len(current_df),
            "drift_results":
                drift_results,
            "created_at":
                datetime.now().isoformat(),
        }

    finally:

        conn.close()


if __name__ == "__main__":

    result = run_drift_check()

    print(result)