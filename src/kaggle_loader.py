import os
from datetime import datetime

import pandas as pd

from src.database import get_connection, init_db


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

TRAIN_PATH = "data/train.csv"
FEATURES_PATH = "data/features.csv"
STORES_PATH = "data/stores.csv"


# ---------------------------------------------------------
# Prototype assumptions
# ---------------------------------------------------------
#
# IMPORTANT:
# The original Walmart Store Sales dataset contains historical
# sales and store-related information, but it does not contain
# direct product-level inventory, expiry, or measured food-waste
# variables.
#
# Therefore, the inventory and waste-related fields created in
# this loader are explicitly treated as SIMULATED / ENGINEERED
# decision-support variables.
#
# They should not be interpreted as observed Walmart food-waste
# measurements.
# ---------------------------------------------------------

ASSUMED_UNIT_PRICE = 10.0

CRITICAL_EXPIRY_DAYS = 2
SHORT_EXPIRY_DAYS = 5

CRITICAL_EXPIRY_WASTE_RATE = 0.40
SHORT_EXPIRY_WASTE_RATE = 0.20
NORMAL_EXPIRY_WASTE_RATE = 0.10


def validate_input_files():
    """
    Check that all required Walmart dataset files are available.
    """

    required_files = {
        "training data": TRAIN_PATH,
        "feature data": FEATURES_PATH,
        "store data": STORES_PATH,
    }

    missing_files = [
        path
        for path in required_files.values()
        if not os.path.exists(path)
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required dataset file(s): "
            + ", ".join(missing_files)
        )


def estimate_waste_quantity(
    current_stock,
    units_sold,
    expiry_days
):
    """
    Estimate potential waste quantity using transparent
    rule-based assumptions.

    The calculation is based on inventory surplus and simulated
    remaining shelf life.

    This value represents a prototype waste-risk estimate for
    decision support. It is NOT measured food waste from the
    original Walmart dataset.
    """

    overstock = (
        current_stock - units_sold
    ).clip(lower=0)

    waste_quantity = pd.Series(
        0.0,
        index=current_stock.index
    )

    # Products very close to simulated expiry are assigned
    # the highest proportion of potential waste.
    critical_mask = (
        expiry_days <= CRITICAL_EXPIRY_DAYS
    )

    waste_quantity.loc[critical_mask] = (
        overstock.loc[critical_mask]
        * CRITICAL_EXPIRY_WASTE_RATE
    )

    # Products with moderately short simulated shelf life
    # receive a lower estimated waste proportion.
    short_expiry_mask = (
        (expiry_days > CRITICAL_EXPIRY_DAYS)
        & (expiry_days <= SHORT_EXPIRY_DAYS)
    )

    waste_quantity.loc[short_expiry_mask] = (
        overstock.loc[short_expiry_mask]
        * SHORT_EXPIRY_WASTE_RATE
    )

    # Products with longer simulated shelf life retain a
    # small baseline potential-waste estimate.
    normal_mask = (
        expiry_days > SHORT_EXPIRY_DAYS
    )

    waste_quantity.loc[normal_mask] = (
        overstock.loc[normal_mask]
        * NORMAL_EXPIRY_WASTE_RATE
    )

    return waste_quantity.round().astype(int)


def load_walmart_data():
    """
    Load and merge the Walmart Store Sales dataset, construct
    demand-related variables, and enrich the records with
    simulated inventory and waste-risk variables for the
    decision-support prototype.
    """

    # Initialize database and required tables.
    init_db()

    # Confirm that all source files are available.
    validate_input_files()

    # -----------------------------------------------------
    # 1. Load original Walmart files
    # -----------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    stores_df = pd.read_csv(STORES_PATH)

    if train_df.empty:
        raise ValueError(
            "The Walmart training dataset is empty."
        )

    if features_df.empty:
        raise ValueError(
            "The Walmart feature dataset is empty."
        )

    if stores_df.empty:
        raise ValueError(
            "The Walmart store dataset is empty."
        )

    # -----------------------------------------------------
    # 2. Merge Walmart datasets
    # -----------------------------------------------------

    df = train_df.merge(
        features_df,
        on=[
            "Store",
            "Date",
            "IsHoliday"
        ],
        how="left"
    )

    df = df.merge(
        stores_df,
        on="Store",
        how="left"
    )

    # -----------------------------------------------------
    # 3. Date and calendar features
    # -----------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    if df["Date"].isna().all():
        raise ValueError(
            "Date conversion failed for the Walmart dataset."
        )

    df["day_of_week"] = (
        df["Date"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    # -----------------------------------------------------
    # 4. Product / department information
    # -----------------------------------------------------

    df["product_name"] = (
        "Dept_" + df["Dept"].astype(str)
    )

    df["category"] = (
        df["Type"]
        .fillna("Unknown")
        .astype(str)
    )

    df["store_id"] = (
        pd.to_numeric(
            df["Store"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df["date"] = (
        df["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    df["is_holiday"] = (
        df["IsHoliday"]
        .fillna(False)
        .astype(int)
    )

    # -----------------------------------------------------
    # 5. Promotion indicator
    # -----------------------------------------------------
    #
    # Walmart provides MarkDown fields rather than a single
    # promotion flag. Therefore, a promotion indicator is
    # created when at least one MarkDown field is positive.
    # -----------------------------------------------------

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    for col in markdown_columns:

        if col not in df.columns:
            df[col] = 0

        df[col] = (
            pd.to_numeric(
                df[col],
                errors="coerce"
            )
            .fillna(0)
        )

    df["promotion"] = (
        (df["MarkDown1"] > 0)
        | (df["MarkDown2"] > 0)
        | (df["MarkDown3"] > 0)
        | (df["MarkDown4"] > 0)
        | (df["MarkDown5"] > 0)
    ).astype(int)

    # -----------------------------------------------------
    # 6. Temperature
    # -----------------------------------------------------

    df["temperature"] = pd.to_numeric(
        df["Temperature"],
        errors="coerce"
    )

    df["temperature"] = (
        df["temperature"]
        .fillna(
            df["temperature"].median()
        )
    )

    # -----------------------------------------------------
    # 7. Approximate demand units
    # -----------------------------------------------------
    #
    # Weekly_Sales is a monetary sales value rather than a
    # direct physical unit count.
    #
    # For this prototype, an assumed unit price of 10 is used
    # to transform sales value into an approximate unit-demand
    # variable:
    #
    # approximate units sold = Weekly_Sales / assumed unit price
    #
    # This is a modelling assumption and must not be interpreted
    # as observed product-unit sales from Walmart.
    # -----------------------------------------------------

    weekly_sales = pd.to_numeric(
        df["Weekly_Sales"],
        errors="coerce"
    ).fillna(0)

    df["units_sold"] = (
        weekly_sales
        .clip(lower=0)
        .div(ASSUMED_UNIT_PRICE)
        .round()
        .astype(int)
    )

    # Constant assumed price used consistently with the
    # approximate units-sold transformation above.
    df["unit_price"] = ASSUMED_UNIT_PRICE

    # -----------------------------------------------------
    # 8. Simulated remaining shelf life
    # -----------------------------------------------------
    #
    # The Walmart dataset does not contain expiry dates.
    #
    # Therefore, expiry_days is explicitly simulated using
    # department identifiers to provide deterministic and
    # reproducible shelf-life variation for the prototype.
    # -----------------------------------------------------

    department_numeric = pd.to_numeric(
        df["Dept"],
        errors="coerce"
    ).fillna(0).astype(int)

    df["expiry_days"] = (
        10 - (department_numeric % 10)
    ).clip(
        lower=1,
        upper=10
    )

    # -----------------------------------------------------
    # 9. Simulated inventory level
    # -----------------------------------------------------
    #
    # The original Walmart dataset does not provide current
    # product-level stock quantities.
    #
    # A simple inventory buffer is therefore added above
    # approximate demand.
    #
    # The buffer increases during business conditions that
    # could reasonably influence stocking decisions:
    # holidays, promotions, and weekends.
    # -----------------------------------------------------

    stock_buffer = (
        30
        + df["is_holiday"] * 20
        + df["promotion"] * 15
        + df["is_weekend"] * 10
    )

    df["current_stock"] = (
        df["units_sold"]
        + stock_buffer
    ).round().astype(int)

    # -----------------------------------------------------
    # 10. Estimated potential waste
    # -----------------------------------------------------
    #
    # Food waste is NOT directly observed in the Walmart data.
    #
    # Potential waste is estimated from:
    #
    #     inventory surplus
    #             +
    #     simulated expiry condition
    #
    # Shorter shelf life produces a higher waste-risk estimate.
    #
    # The resulting variable is used only as an engineered
    # decision-support indicator for the prototype.
    # -----------------------------------------------------

    df["waste_quantity"] = (
        estimate_waste_quantity(
            current_stock=df["current_stock"],
            units_sold=df["units_sold"],
            expiry_days=df["expiry_days"]
        )
    )

    # -----------------------------------------------------
    # 11. Data provenance
    # -----------------------------------------------------
    #
    # Clearly identify that the original Walmart data has
    # been enriched with simulated inventory/waste variables.
    # -----------------------------------------------------

    df["source"] = (
        "walmart_kaggle_dataset_"
        "with_simulated_inventory_"
        "and_waste_risk_features"
    )

    df["created_at"] = (
        datetime.now().isoformat()
    )

    # -----------------------------------------------------
    # 12. Final dataset
    # -----------------------------------------------------

    final_columns = [
        "product_name",
        "category",
        "store_id",
        "date",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "promotion",
        "temperature",
        "current_stock",
        "units_sold",
        "unit_price",
        "expiry_days",
        "waste_quantity",
        "source",
        "created_at"
    ]

    final_df = df[
        final_columns
    ].copy()

    # Final validation before database insertion.
    if final_df.empty:
        raise ValueError(
            "No records were produced after processing."
        )

    # -----------------------------------------------------
    # 13. Store processed records in SQLite
    # -----------------------------------------------------

    conn = get_connection()

    try:

        # Replace previously loaded historical records.
        conn.execute(
            "DELETE FROM raw_food_sales"
        )

        final_df.to_sql(
            "raw_food_sales",
            conn,
            if_exists="append",
            index=False
        )

        conn.commit()

    finally:
        conn.close()

    # -----------------------------------------------------
    # Completion summary
    # -----------------------------------------------------

    print(
        "Walmart Kaggle data loaded successfully."
    )

    print(
        "Rows inserted:",
        len(final_df)
    )

    print(
        "Demand variable created from Weekly_Sales "
        "using an assumed unit price."
    )

    print(
        "Simulated inventory and expiry features "
        "generated successfully."
    )

    print(
        "Potential waste quantity estimated using "
        "inventory surplus and shelf-life assumptions."
    )

    print(
        "IMPORTANT: waste_quantity represents an "
        "engineered decision-support estimate, "
        "not observed Walmart food waste."
    )

    return final_df


if __name__ == "__main__":
    load_walmart_data()