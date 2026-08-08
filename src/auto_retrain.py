from datetime import datetime

from src.drift_detection import run_drift_check
from src.logging_utils import append_csv_log
from src.train import train_model


RETRAINING_LOG_PATH = "logs/retraining_log.csv"


def build_log_row(
    status,
    reason,
    metrics=None,
    error_message=None,
):

    metrics = metrics or {}

    return {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "reason": reason,
        "model_name": metrics.get("model_name"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "train_rmse": metrics.get("train_rmse"),
        "cv_rmse_mean": metrics.get("cv_rmse_mean"),
        "generalization_gap": metrics.get(
            "generalization_gap"
        ),
        "training_rows": metrics.get("training_rows"),
        "testing_rows": metrics.get("testing_rows"),
        "model_path": metrics.get("model_path"),
        "error_message": error_message,
    }


def auto_retrain():

    drift_result = run_drift_check()

    if not drift_result.get(
        "drift_detected",
        False,
    ):

        print(
            "No drift detected. Retraining skipped."
        )

        log_row = build_log_row(
            status="skipped",
            reason="no_drift_detected",
        )

        append_csv_log(
            RETRAINING_LOG_PATH,
            log_row,
        )

        return {
            "status": "skipped",
            "reason": "no_drift_detected",
            "drifted_features": drift_result.get(
                "drifted_features",
                [],
            ),
            "drift_result": drift_result,
        }

    print(
        "Drift detected. Starting automatic retraining..."
    )

    try:

        metrics = train_model()

        log_row = build_log_row(
            status="retrained",
            reason="drift_detected",
            metrics=metrics,
        )

        append_csv_log(
            RETRAINING_LOG_PATH,
            log_row,
        )

        return {
            "status": "retrained",
            "reason": "drift_detected",
            "drifted_features": drift_result.get(
                "drifted_features",
                [],
            ),
            "metrics": metrics,
            "drift_result": drift_result,
        }

    except Exception as error:

        error_message = str(error)

        print(
            "Automatic retraining failed:",
            error_message,
        )

        log_row = build_log_row(
            status="failed",
            reason="retraining_error",
            error_message=error_message,
        )

        append_csv_log(
            RETRAINING_LOG_PATH,
            log_row,
        )

        return {
            "status": "failed",
            "reason": "retraining_error",
            "error": error_message,
            "drift_result": drift_result,
        }


if __name__ == "__main__":

    result = auto_retrain()

    print(result)