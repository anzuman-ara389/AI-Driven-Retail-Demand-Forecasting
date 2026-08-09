import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title=(
        "AI-Driven Retail Demand "
        "Forecasting Dashboard"
    ),
    layout="wide",
)


st.title(
    "AI-Driven Retail Demand Forecasting "
    "and Intelligent Inventory Decision Support"
)

st.write(
    "Predict demand, assess inventory, classify waste risk, "
    "monitor drift, compare models, and trigger retraining."
)


def api_get(endpoint):

    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=30,
        )

        return response

    except requests.exceptions.RequestException as error:

        st.error(
            f"API connection error: {error}"
        )

        return None


def api_post(
    endpoint,
    payload=None,
    timeout=180,
):

    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=timeout,
        )

        return response

    except requests.exceptions.RequestException as error:

        st.error(
            f"API connection error: {error}"
        )

        return None


def show_response_error(response):

    if response is None:
        return

    try:
        error_data = response.json()

        if isinstance(error_data, dict):

            message = (
                error_data.get("detail")
                or error_data.get("error")
                or error_data
            )

        else:
            message = error_data

    except ValueError:

        message = response.text

    st.error(
        f"Request failed: {message}"
    )


health_response = api_get(
    "/health"
)

with st.sidebar:

    st.header(
        "System Status"
    )

    if (
        health_response is not None
        and health_response.status_code == 200
    ):

        health_data = health_response.json()

        st.success(
            "API Connected"
        )

        if health_data.get(
            "model_available"
        ):

            st.info(
                "Model Available"
            )

        else:

            st.warning(
                "Model Not Available"
            )

    else:

        st.error(
            "API Not Connected"
        )


menu = st.sidebar.selectbox(
    "Select Page",
    [
        "Predict Demand",
        "Latest Sales",
        "Prediction Logs",
        "Model Info",
        "Model Comparison",
        "Drift Reports",
        "Pipeline Actions",
    ],
)


if menu == "Predict Demand":

    st.header(
        "Demand Prediction and Inventory Assessment"
    )

    col1, col2 = st.columns(
        2
    )

    with col1:

        product_name = st.selectbox(
            "Department",
            [
                "Dept_1",
                "Dept_2",
                "Dept_3",
                "Dept_4",
                "Dept_5",
                "Dept_6",
            ],
        )

        category = st.selectbox(
            "Store Category",
            [
                "A",
                "B",
                "C",
                "General",
            ],
        )

        store_id = st.number_input(
            "Store ID",
            min_value=1,
            max_value=45,
            value=1,
            step=1,
        )

        day_of_week = st.selectbox(
            "Day of Week",
            options=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            format_func=lambda value: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ][value],
        )

        is_weekend = (
            1
            if day_of_week in [5, 6]
            else 0
        )

        is_holiday = st.selectbox(
            "Holiday",
            options=[
                0,
                1,
            ],
            format_func=lambda value: (
                "Yes"
                if value == 1
                else "No"
            ),
        )

    with col2:

        promotion = st.selectbox(
            "Promotion",
            options=[
                0,
                1,
            ],
            format_func=lambda value: (
                "Active"
                if value == 1
                else "Inactive"
            ),
        )

        temperature = st.number_input(
            "Temperature",
            value=60.0,
            step=1.0,
        )

        current_stock = st.number_input(
            "Current Stock",
            min_value=0,
            value=100,
            step=1,
        )

        unit_price = st.number_input(
            "Assumed Unit Price",
            min_value=0.01,
            value=10.0,
            step=0.50,
        )

        expiry_days = st.number_input(
            "Remaining Shelf-Life Days",
            min_value=0,
            value=5,
            step=1,
        )

    predict_button = st.button(
        "Predict Demand",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        payload = {
            "product_name":
                product_name,
            "category":
                category,
            "store_id":
                int(store_id),
            "day_of_week":
                int(day_of_week),
            "is_weekend":
                int(is_weekend),
            "is_holiday":
                int(is_holiday),
            "promotion":
                int(promotion),
            "temperature":
                float(temperature),
            "current_stock":
                int(current_stock),
            "unit_price":
                float(unit_price),
            "expiry_days":
                int(expiry_days),
        }

        with st.spinner(
            "Generating prediction..."
        ):

            response = api_post(
                "/predict-demand",
                payload=payload,
                timeout=120,
            )

        if (
            response is not None
            and response.status_code == 200
        ):

            result = response.json()

            st.success(
                "Prediction completed successfully."
            )

            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )

            metric1.metric(
                "Predicted Demand",
                result.get(
                    "predicted_demand",
                    0,
                ),
            )

            metric2.metric(
                "Current Stock",
                result.get(
                    "current_stock",
                    0,
                ),
            )

            metric3.metric(
                "Inventory Status",
                result.get(
                    "inventory_status",
                    "Unknown",
                ),
            )

            metric4.metric(
                "Waste Risk",
                result.get(
                    "waste_risk",
                    "Unknown",
                ),
            )

            metric5, metric6, metric7, metric8 = (
                st.columns(4)
            )

            metric5.metric(
                "Expiry Status",
                result.get(
                    "expiry_status",
                    "Unknown",
                ),
            )

            metric6.metric(
                "Inventory Gap",
                result.get(
                    "inventory_gap",
                    0,
                ),
            )

            metric7.metric(
                "Safety Stock",
                result.get(
                    "safety_stock",
                    0,
                ),
            )

            metric8.metric(
                "Recommended Order",
                result.get(
                    "recommended_order_quantity",
                    0,
                ),
            )

            st.subheader(
                "Suggested Business Action"
            )

            st.info(
                result.get(
                    "suggested_action",
                    "No action available.",
                )
            )

            st.subheader(
                "LLM-Powered Intelligent Recommendation"
            )

            st.success(
                result.get(
                    "recommendation",
                    "No recommendation available.",
                )
            )

            recommendation_source = result.get(
                "recommendation_source",
                "unknown",
            )

            llm_model = result.get(
                "llm_model",
                "not_available",
            )

            source_col, model_col = st.columns(
                2
            )

            source_col.metric(
                "Recommendation Source",
                recommendation_source,
            )

            model_col.metric(
                "LLM Model",
                llm_model,
            )

            if recommendation_source == "local_llm_qwen2.5":

                st.success(
                    "Local LLM recommendation generated successfully."
                )

            else:

                st.warning(
                    "Fallback recommendation used because the local LLM "
                    "was not available."
                )

            with st.expander(
                "View complete API response"
            ):

                st.json(
                    result
                )

        else:

            show_response_error(
                response
            )


elif menu == "Latest Sales":

    st.header(
        "Latest Sales Records"
    )

    limit = st.slider(
        "Number of records",
        min_value=5,
        max_value=100,
        value=20,
    )

    response = api_get(
        f"/latest-sales?limit={limit}"
    )

    if (
        response is not None
        and response.status_code == 200
    ):

        data = response.json()

        if not data:

            st.info(
                "No sales records found."
            )

        else:

            sales_df = pd.DataFrame(
                data
            )

            st.dataframe(
                sales_df,
                use_container_width=True,
                hide_index=True,
            )

    else:

        show_response_error(
            response
        )


elif menu == "Prediction Logs":

    st.header(
        "Prediction Logs"
    )

    limit = st.slider(
        "Number of prediction logs",
        min_value=5,
        max_value=100,
        value=20,
    )

    response = api_get(
        f"/prediction-logs?limit={limit}"
    )

    if (
        response is not None
        and response.status_code == 200
    ):

        data = response.json()

        if not data:

            st.info(
                "No prediction logs found."
            )

        else:

            logs_df = pd.DataFrame(
                data
            )

            st.dataframe(
                logs_df,
                use_container_width=True,
                hide_index=True,
            )

    else:

        show_response_error(
            response
        )


elif menu == "Model Info":

    st.header(
        "Latest Registered Model"
    )

    response = api_get(
        "/model-info"
    )

    if (
        response is not None
        and response.status_code == 200
    ):

        result = response.json()

        if "message" in result:

            st.info(
                result["message"]
            )

        else:

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Best Model",
                result.get(
                    "model_name",
                    "Unknown",
                ),
            )

            col2.metric(
                "Test MAE",
                round(
                    float(
                        result.get(
                            "mae",
                            0,
                        )
                    ),
                    4,
                ),
            )

            col3.metric(
                "Test RMSE",
                round(
                    float(
                        result.get(
                            "rmse",
                            0,
                        )
                    ),
                    4,
                ),
            )

            col4.metric(
                "Test R²",
                round(
                    float(
                        result.get(
                            "r2",
                            0,
                        )
                    ),
                    4,
                ),
            )

            st.json(
                result
            )

    else:

        show_response_error(
            response
        )


elif menu == "Model Comparison":

    st.header(
        "Random Forest vs XGBoost"
    )

    response = api_get(
        "/model-comparison"
    )

    if (
        response is not None
        and response.status_code == 200
    ):

        data = response.json()

        if isinstance(
            data,
            dict,
        ):

            st.info(
                data.get(
                    "message",
                    "No comparison data available.",
                )
            )

        else:

            comparison_df = pd.DataFrame(
                data
            )

            if comparison_df.empty:

                st.info(
                    "No model comparison found."
                )

            else:

                display_columns = [
                    column
                    for column in [
                        "model_name",
                        "train_rmse",
                        "rmse",
                        "r2",
                        "cv_rmse_mean",
                        "cv_rmse_std",
                        "generalization_gap",
                        "absolute_gap",
                    ]
                    if column
                    in comparison_df.columns
                ]

                st.dataframe(
                    comparison_df[
                        display_columns
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                selection_column = (
                    "cv_rmse_mean"
                    if "cv_rmse_mean"
                    in comparison_df.columns
                    else "rmse"
                )

                best_model = (
                    comparison_df.sort_values(
                        selection_column
                    )
                    .iloc[0]
                )

                st.success(
                    "Best Model: "
                    f"{best_model['model_name']} "
                    f"(selected using {selection_column})"
                )

                if (
                    "cv_rmse_mean"
                    in comparison_df.columns
                ):

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    col1.metric(
                        "Best CV RMSE",
                        round(
                            float(
                                best_model[
                                    "cv_rmse_mean"
                                ]
                            ),
                            4,
                        ),
                    )

                    col2.metric(
                        "Test RMSE",
                        round(
                            float(
                                best_model[
                                    "rmse"
                                ]
                            ),
                            4,
                        ),
                    )

                    col3.metric(
                        "Test R²",
                        round(
                            float(
                                best_model[
                                    "r2"
                                ]
                            ),
                            4,
                        ),
                    )

    else:

        show_response_error(
            response
        )


elif menu == "Drift Reports":

    st.header(
        "Drift Monitoring Reports"
    )

    limit = st.slider(
        "Number of drift records",
        min_value=5,
        max_value=100,
        value=20,
    )

    response = api_get(
        f"/drift-reports?limit={limit}"
    )

    if (
        response is not None
        and response.status_code == 200
    ):

        data = response.json()

        if not data:

            st.info(
                "No drift reports found."
            )

        else:

            drift_df = pd.DataFrame(
                data
            )

            st.dataframe(
                drift_df,
                use_container_width=True,
                hide_index=True,
            )

            if (
                "drift_detected"
                in drift_df.columns
            ):

                detected_count = int(
                    drift_df[
                        "drift_detected"
                    ].sum()
                )

                st.metric(
                    "Drifted Records",
                    detected_count,
                )

    else:

        show_response_error(
            response
        )


elif menu == "Pipeline Actions":

    st.header(
        "Pipeline Controls"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        if st.button(
            "Run Full Pipeline",
            use_container_width=True,
        ):

            with st.spinner(
                "Running preprocessing and training..."
            ):

                response = api_post(
                    "/run-pipeline",
                    timeout=600,
                )

            if (
                response is not None
                and response.status_code == 200
            ):

                st.success(
                    "Pipeline completed."
                )

                st.json(
                    response.json()
                )

            else:

                show_response_error(
                    response
                )

    with col2:

        if st.button(
            "Run Drift Check",
            use_container_width=True,
        ):

            with st.spinner(
                "Checking drift..."
            ):

                response = api_post(
                    "/drift-check",
                    timeout=180,
                )

            if (
                response is not None
                and response.status_code == 200
            ):

                result = response.json()

                if result.get(
                    "drift_detected"
                ):

                    st.warning(
                        "Drift detected in: "
                        + ", ".join(
                            result.get(
                                "drifted_features",
                                [],
                            )
                        )
                    )

                else:

                    st.success(
                        "No drift detected."
                    )

                st.json(
                    result
                )

            else:

                show_response_error(
                    response
                )

    with col3:

        if st.button(
            "Auto Retrain",
            use_container_width=True,
        ):

            with st.spinner(
                "Checking drift and retraining if required..."
            ):

                response = api_post(
                    "/auto-retrain",
                    timeout=900,
                )

            if (
                response is not None
                and response.status_code == 200
            ):

                result = response.json()

                status = result.get(
                    "status"
                )

                if status == "retrained":

                    st.success(
                        "Drift detected and model retrained."
                    )

                elif status == "skipped":

                    st.info(
                        "No drift detected. Retraining skipped."
                    )

                else:

                    st.error(
                        "Automatic retraining failed."
                    )

                st.json(
                    result
                )

            else:

                show_response_error(
                    response
                )

    st.divider()

    st.subheader(
        "Live Data Simulation Commands"
    )

    st.info(
        "Normal live events:\n\n"
        "`python -m src.external_client "
        "--mode normal --events 20 --delay 1`"
    )

    st.warning(
        "Shifted drift events:\n\n"
        "`python -m src.external_client "
        "--mode shifted --events 300 --delay 0.2`"
    )