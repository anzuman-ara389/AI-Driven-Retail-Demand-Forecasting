# MSc BDS -2nd Semester Project
# AI-Driven Retail Demand Forecasting and Intelligent Inventory Decision Support for Food Waste Reduction

## Overview

This project implements an end-to-end retail demand forecasting and inventory decision-support system. It combines machine-learning demand forecasting, inventory and expiry assessment, inventory-related waste-risk identification, drift monitoring, automatic retraining, and a local LLM recommendation layer.

The system uses FastAPI for API-based ingestion and prediction, SQLite for storage, Streamlit for the decision-support dashboard, and Ollama with Qwen2.5:1.5b for context-aware inventory recommendations.

## Project Architecture

```text
Simulated Retail Data
        ↓
FastAPI
        ↓
SQLite
        ↓
Preprocessing & Feature Engineering
        ↓
Model Training / Demand Prediction
        ↓
Inventory & Waste-Risk Assessment
        ↓
Rule-Based Business Action
        ↓
Local LLM Recommendation
(Ollama + Qwen2.5:1.5b)
        ↓
Prediction Logs / Drift Detection
        ↓
Automatic Retraining
        ↓
Streamlit Dashboard
```

## Data and Scope

The project uses the Walmart Store Sales Forecasting dataset as the primary historical demand dataset.

Inventory, expiry, and waste-related variables are engineered or simulated because the original dataset does not contain verified physical food-waste measurements. Therefore, the system estimates **inventory-related waste risk** rather than claiming to measure Walmart's actual discarded food waste.

## Machine Learning

The current implementation compares:

- `RandomForestRegressor`
- `XGBoostRegressor`

Model evaluation includes:

- MAE
- RMSE
- R²
- Train/test performance
- Time-series cross-validation
- Generalization gap

`TimeSeriesSplit` is used for cross-validation to evaluate model generalization while respecting the temporal nature of the demand data.

## Prediction and Inventory Decision Support

The main prediction endpoint is:

```text
POST /predict-demand
```

The prediction workflow:

1. Receives and validates retail input.
2. Builds the required model features.
3. Predicts demand using the trained ML model.
4. Compares predicted demand with current stock.
5. Assesses inventory and expiry status.
6. Classifies inventory-related waste risk.
7. Generates a rule-based business action.
8. Sends the structured results to the local LLM.
9. Returns the prediction and recommendation to the dashboard.

The decision-support output includes predicted demand, current stock, inventory status, inventory gap, safety stock, recommended order quantity, expiry status, waste risk, and suggested business action.

## Food Waste-Risk Interpretation

The project does **not** calculate verified real-world discarded food quantities. Instead, it uses engineered inventory and expiry information to identify conditions associated with higher waste risk, such as excess stock relative to expected demand and short remaining shelf-life.

The purpose is preventive decision support: helping managers take actions that may reduce overstock, expiry, and inventory-related food waste.

## Local LLM Recommendation

The recommendation layer uses:

- **Ollama** — local LLM runtime
- **Qwen2.5:1.5b** — local language model

The LLM **does not predict demand**. Demand prediction is performed by the trained machine-learning model.

The LLM receives structured information such as predicted demand, current stock, inventory status, expiry status, waste risk, and the rule-based action. It converts these results into a contextual recommendation for the retail decision maker.

If the local LLM is unavailable, the deterministic rule-based recommendation can be used as a fallback.

### Local LLM Setup

```bash
ollama pull qwen2.5:1.5b
```

Check the model:

```bash
ollama list
```

No paid OpenAI API key is required for the local Qwen implementation.

## Drift Detection and Retraining

The system monitors incoming data for drift. The implementation compares a reference window with recent observations and uses a configured drift threshold of `0.15`.

Main endpoints:

```text
POST /drift-check
POST /auto-retrain
```

When significant drift is detected, the retraining component can re-run model training, evaluate candidate models, and update the production model and its artifacts.

## Streamlit Dashboard

The Streamlit dashboard provides access to:

- Demand prediction and inventory assessment
- LLM-powered intelligent recommendations
- Latest sales
- Prediction logs
- Model information and comparison
- Drift reports
- Pipeline actions

## Main Project Structure

```text
project/
├── .github/workflows/
├── artifacts/
├── logs/
├── report/
├── src/
│   ├── auto_retrain.py
│   ├── database.py
│   ├── drift_detection.py
│   ├── external_client.py
│   ├── frontend.py
│   ├── inventory_assessment.py
│   ├── kaggle_loader.py
│   ├── live_ingestion.py
│   ├── llm_recommendation.py
│   ├── logging_utils.py
│   ├── main.py
│   ├── preprocess.py
│   └── train.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.train
```

Start FastAPI:

```bash
uvicorn src.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

In another terminal, start Streamlit:

```bash
streamlit run src/frontend.py
```

Dashboard:

```text
http://localhost:8501
```

## Main API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check API/model status |
| `/sales-event` | POST | Receive sales data |
| `/predict-demand` | POST | Predict demand and generate decision support |
| `/prediction-logs` | GET | View prediction history |
| `/run-pipeline` | POST | Run the ML pipeline |
| `/model-info` | GET | View model information |
| `/model-comparison` | GET | View model comparison |
| `/drift-check` | POST | Run drift detection |
| `/drift-reports` | GET | View drift reports |
| `/auto-retrain` | POST | Run automatic retraining logic |

## Testing

Run automated tests:

```bash
pytest tests/ -v
```

Check Python syntax:

```bash
python -m compileall src
```

The repository also includes a GitHub Actions workflow for automated pipeline validation.

## Technology Stack

Python, pandas, NumPy, scikit-learn, XGBoost, FastAPI, Pydantic, SQLite, Streamlit, joblib, requests, Ollama, Qwen2.5:1.5b, GitHub Actions, and Docker.

## Conclusion

This project demonstrates an end-to-end retail decision-support pipeline that combines machine-learning demand forecasting with inventory assessment, inventory-related waste-risk identification, monitoring and retraining, and local LLM-powered recommendations. The LLM complements the forecasting model by translating structured prediction and inventory results into context-aware business recommendations.
