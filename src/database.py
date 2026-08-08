import os
import sqlite3

DB_PATH = "data/food_waste.db"


def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():

    os.makedirs("data", exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_food_sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        category TEXT,
        store_id INTEGER,
        date TEXT,
        day_of_week INTEGER,
        is_weekend INTEGER,
        is_holiday INTEGER,
        promotion INTEGER,
        temperature REAL,
        current_stock INTEGER,
        units_sold INTEGER,
        unit_price REAL,
        expiry_days INTEGER,
        waste_quantity INTEGER,
        source TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_demand REAL,
    current_stock INTEGER,
    inventory_status TEXT,
    expiry_status TEXT,
    recommendation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS drift_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT,
        reference_mean REAL,
        current_mean REAL,
        drift_score REAL,
        drift_detected INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        model_path TEXT,
        mae REAL,
        rmse REAL,
        r2 REAL,
        training_rows INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()