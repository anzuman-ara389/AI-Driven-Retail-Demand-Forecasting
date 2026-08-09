import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.auto_retrain import auto_retrain
from src.database import get_connection, init_db
from src.drift_detection import run_drift_check
from src.inventory_assessment import assess_inventory
from src.llm_recommendation import generate_recommendation
from src.preprocess import preprocess_data
from src.train import FEATURE_COLUMNS, MODEL_PATH, train_model


app = FastAPI(
    title=(
        "AI-Driven Retail Demand Forecasting and "
        "Intelligent Inventory Decision Support API"
    ),
    description=(
        "Demand forecasting, inventory assessment, "
        "waste-risk classification, local LLM recommendation, "
        "monitoring, and retraining."
    ),
    version="2.2.0",
)


class SalesEvent(BaseModel):
    product_name: str = "Dept_1"
    category: str = "A"

    store_id: int = Field(
        default=1,
        ge=1,
        le=45,
    )

    date: str = Field(
        default_factory=lambda: datetime.now().strftime(
            "%Y-%m-%d"
        )
    )

    day_of_week: int = Field(
        default_factory=lambda: datetime.now().weekday(),
        ge=0,
        le=6,
    )

    is_weekend: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    is_holiday: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    promotion: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    temperature: float = 20.0

    current_stock: int = Field(
        default=100,
        ge=0,
    )

    units_sold: int = Field(
        default=60,
        ge=0,
    )

    unit_price: float = Field(
        default=10.0,
        gt=0,
    )

    expiry_days: int = Field(
        default=5,
        ge=0,
    )

    waste_quantity: int = Field(
        default=0,
        ge=0,
    )

    source: str = "api_event"


class PredictionRequest(BaseModel):
    product_name: str = "Dept_1"
    category: str = "A"

    store_id: int = Field(
        default=1,
        ge=1,
        le=45,
    )

    day_of_week: int = Field(
        default_factory=lambda: datetime.now().weekday(),
        ge=0,
        le=6,
    )

    is_weekend: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    is_holiday: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    promotion: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    temperature: float = 20.0

    current_stock: int = Field(
        default=100,
        ge=0,
    )

    unit_price: float = Field(
        default=10.0,
        gt=0,
    )

    expiry_days: int = Field(
        default=5,
        ge=0,
    )


@app.on_event("startup")
def startup_event():
    init_db()


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


def load_encoding_maps():
    processed_path = "data/processed_food_sales.csv"

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            "Processed dataset not found. "
            "Run preprocessing first."
        )

    df = pd.read_csv(
        processed_path,
        usecols=[
            "product_name",
            "product_encoded",
            "category",
            "category_encoded",
        ],
    )

    product_map = (
        df[
            [
                "product_name",
                "product_encoded",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "product_name"
        )["product_encoded"]
        .to_dict()
    )

    category_map = (
        df[
            [
                "category",
                "category_encoded",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "category"
        )["category_encoded"]
        .to_dict()
    )

    return product_map, category_map


def encode_product(product_name):
    try:
        product_map, _ = load_encoding_maps()

        return int(
            product_map.get(
                product_name,
                0,
            )
        )

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
    ):
        return 0


def encode_category(category):
    try:
        _, category_map = load_encoding_maps()

        return int(
            category_map.get(
                category,
                0,
            )
        )

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
    ):
        return 0


def build_prediction_features(data):

    short_expiry = int(
        data.expiry_days <= 3
    )

    promotion_active = int(
        data.promotion
    )

    row = {
        "product_encoded": encode_product(
            data.product_name
        ),
        "category_encoded": encode_category(
            data.category
        ),
        "store_id": int(
            data.store_id
        ),
        "day_of_week": int(
            data.day_of_week
        ),
        "is_weekend": int(
            data.is_weekend
        ),
        "is_holiday": int(
            data.is_holiday
        ),
        "promotion": int(
            data.promotion
        ),
        "temperature": float(
            data.temperature
        ),
        "unit_price": float(
            data.unit_price
        ),
        "expiry_days": int(
            data.expiry_days
        ),
        "short_expiry": short_expiry,
        "promotion_active": promotion_active,
    }

    input_df = pd.DataFrame(
        [row]
    )

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: "
            f"{missing_columns}"
        )

    return input_df[
        FEATURE_COLUMNS
    ]


@app.get("/")
def root():

    return {
        "message": (
            "Retail demand forecasting and "
            "intelligent inventory decision-support API"
        ),
        "version": "2.2.0",
        "llm": "Qwen2.5:1.5B via Ollama",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_available": os.path.exists(
            MODEL_PATH
        ),
        "llm_model": "qwen2.5:1.5b",
    }


@app.post("/sales-event")
def receive_sales_event(
    event: SalesEvent
):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO raw_food_sales (
                product_name,
                category,
                store_id,
                date,
                day_of_week,
                is_weekend,
                is_holiday,
                promotion,
                temperature,
                current_stock,
                units_sold,
                unit_price,
                expiry_days,
                waste_quantity,
                source,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.product_name,
                event.category,
                event.store_id,
                event.date,
                event.day_of_week,
                event.is_weekend,
                event.is_holiday,
                event.promotion,
                event.temperature,
                event.current_stock,
                event.units_sold,
                event.unit_price,
                event.expiry_days,
                event.waste_quantity,
                event.source,
                datetime.now().isoformat(),
            ),
        )

        record_id = cursor.lastrowid

        conn.commit()

    finally:
        conn.close()

    return {
        "status": "success",
        "message": (
            "Sales event stored successfully."
        ),
        "record_id": record_id,
        "source": event.source,
    }


@app.post("/predict-demand")
def predict_demand(
    data: PredictionRequest
):

    model = load_model()

    if model is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No trained model found. "
                "Run python -m src.train first."
            ),
        )

    try:
        input_df = build_prediction_features(
            data
        )

        predicted_demand = float(
            model.predict(
                input_df
            )[0]
        )

        predicted_demand = max(
            0.0,
            predicted_demand,
        )

        inventory = assess_inventory(
            predicted_demand=predicted_demand,
            current_stock=data.current_stock,
            expiry_days=data.expiry_days,
        )

        recommendation_result = generate_recommendation(
            predicted_demand=predicted_demand,
            current_stock=data.current_stock,
            inventory_status=inventory[
                "inventory_status"
            ],
            expiry_status=inventory[
                "expiry_status"
            ],
            promotion=data.promotion,
            holiday=data.is_holiday,
            waste_risk=inventory[
                "waste_risk"
            ],
        )

        recommendation = recommendation_result[
            "recommendation"
        ]

        recommendation_source = recommendation_result[
            "recommendation_source"
        ]

        safety_stock = max(
            5,
            round(
                predicted_demand * 0.10
            ),
        )

        recommended_order_quantity = max(
            0,
            round(
                predicted_demand
                + safety_stock
                - data.current_stock
            ),
        )

        conn = get_connection()

        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO prediction_logs (
                    predicted_demand,
                    current_stock,
                    inventory_status,
                    expiry_status,
                    recommendation,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    predicted_demand,
                    data.current_stock,
                    inventory[
                        "inventory_status"
                    ],
                    inventory[
                        "expiry_status"
                    ],
                    recommendation,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()

        finally:
            conn.close()

        return {
            "predicted_demand": round(
                predicted_demand,
                2,
            ),
            "current_stock": (
                data.current_stock
            ),
            "inventory_gap": inventory[
                "inventory_gap"
            ],
            "inventory_status": inventory[
                "inventory_status"
            ],
            "expiry_status": inventory[
                "expiry_status"
            ],
            "waste_risk": inventory[
                "waste_risk"
            ],
            "suggested_action": inventory[
                "suggested_action"
            ],
            "safety_stock": safety_stock,
            "recommended_order_quantity": (
                recommended_order_quantity
            ),
            "recommendation": recommendation,
            "recommendation_source": (
                recommendation_source
            ),
            "llm_model": "qwen2.5:1.5b",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error


@app.post("/run-pipeline")
def run_pipeline():

    try:
        processed_df = preprocess_data()

        metrics = train_model()

        return {
            "status": "completed",
            "processed_rows": len(
                processed_df
            ),
            "metrics": metrics,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error


@app.post("/drift-check")
def drift_check():

    try:
        return run_drift_check()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error


@app.post("/auto-retrain")
def run_auto_retrain():

    try:
        return auto_retrain()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error


@app.get("/model-info")
def model_info():

    conn = get_connection()

    try:
        query = """
            SELECT *
            FROM model_registry
            ORDER BY id DESC
            LIMIT 1
        """

        df = pd.read_sql_query(
            query,
            conn,
        )

    finally:
        conn.close()

    if df.empty:
        return {
            "message": (
                "No registered model found."
            )
        }

    return df.iloc[
        0
    ].to_dict()


@app.get("/model-comparison")
def model_comparison():

    comparison_path = (
        "artifacts/model_comparison.csv"
    )

    if not os.path.exists(
        comparison_path
    ):
        return {
            "message": (
                "No model comparison found. "
                "Run model training first."
            )
        }

    df = pd.read_csv(
        comparison_path
    )

    return df.to_dict(
        orient="records"
    )


@app.get("/latest-sales")
def latest_sales(
    limit: int = 20
):

    limit = max(
        1,
        min(
            limit,
            100,
        ),
    )

    conn = get_connection()

    try:
        query = """
            SELECT *
            FROM raw_food_sales
            ORDER BY id DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(
                limit,
            ),
        )

    finally:
        conn.close()

    return df.to_dict(
        orient="records"
    )


@app.get("/prediction-logs")
def prediction_logs(
    limit: int = 20
):

    limit = max(
        1,
        min(
            limit,
            100,
        ),
    )

    conn = get_connection()

    try:
        query = """
            SELECT *
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(
                limit,
            ),
        )

    finally:
        conn.close()

    return df.to_dict(
        orient="records"
    )


@app.get("/drift-reports")
def drift_reports(
    limit: int = 20
):

    limit = max(
        1,
        min(
            limit,
            100,
        ),
    )

    conn = get_connection()

    try:
        query = """
            SELECT *
            FROM drift_reports
            ORDER BY id DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(
                limit,
            ),
        )

    finally:
        conn.close()

    return df.to_dict(
        orient="records"
    )