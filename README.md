# INVESTRA

INVESTRA is a gold price forecasting and investor decision-support system that combines time-series modeling, evaluation metrics, and actionable `Buy/Hold/Sell` recommendations.

## Overview

This project started from a research notebook and has been upgraded to a modular Python pipeline with:

- data ingestion and preprocessing
- feature engineering for forecasting
- model training (XGBoost baseline, LSTM, Prophet modules)
- performance evaluation
- API and dashboard layers for serving predictions

### Project Objective

The goal of INVESTRA is to help retail investors understand likely short-term gold price movement and risk before taking action. Instead of only showing a raw prediction, the system is designed to provide:

- a forecasted price path
- quality metrics for model trust
- a recommendation layer (`Buy/Hold/Sell`)
- a simple interface (API + dashboard) for practical usage

### Problem Statement

Gold price is influenced by many factors and often behaves non-linearly. A single static model usually fails across different market regimes. This project addresses that by creating a reusable, testable pipeline that can evolve from pure OHLC modeling to multi-source forecasting with macro and sentiment features.

### Current Scope (Implemented)

- Historical OHLCV ingestion from CSV
- Robust date parsing across multiple vendor formats
- Feature engineering with lags, moving averages, returns, and volatility
- XGBoost baseline training pipeline with leakage-aware validation
- Evaluation with both error metrics and directional performance
- Recommendation logic based on expected ROI and risk
- FastAPI endpoint and Streamlit dashboard starter

### Future Scope

- Macro features (CPI, interest rates, DXY, yields)
- News/sentiment and event signals
- Confidence intervals and calibrated uncertainty
- Ensemble weighting from rolling validation

## Key Features

- **Leakage-aware pipeline**: separates training/validation/testing correctly for model reliability.
- **Decision support output**: recommendation logic based on predicted ROI and uncertainty.
- **Multi-interface usage**: script pipeline, FastAPI service, and Streamlit dashboard.
- **Production-ready structure**: reusable modules instead of notebook-only workflow.

## Repository Structure

```text
INVESTRA/
|- Gold_price_prediction.ipynb
|- requirements.txt
|- .env.example
|- src/
|  |- data_ingestion.py
|  |- features.py
|  |- evaluate.py
|  |- recommendation.py
|  |- pipeline.py
|  |- api.py
|  \- models/
|     |- train_xgb.py
|     |- train_lstm.py
|     |- train_prophet.py
|     \- ensemble.py
\- dashboard/
   \- app.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and update values as needed.

Example:

```env
DATA_PATH=Gold Price Dynamics/GoldUSD.csv
DATA_PATHS=Gold Price Dynamics/GoldUSD.csv,data/XAU_1d_data.csv
MODEL_DIR=./artifacts
FORECAST_FILE=./artifacts/daily_forecast.json
FORECAST_REFRESH_HOURS=24
API_HOST=127.0.0.1
API_PORT=8000
```

## Usage

### 1) Train XGBoost Artifacts

```bash
python -m src.models.train_xgb
```

### 2) Run Full Pipeline (Train + Evaluate)

```bash
python -m src.pipeline
```

### 3) Start API

```bash
uvicorn src.api:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

Auto daily forecast endpoint: `http://127.0.0.1:8000/forecast/latest`

### 4) Start Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard URL: `http://localhost:8501`

## Step-by-Step Run Guide (Recommended)

Use this order if you are running the project from scratch:

1. Install dependencies
2. Set dataset path(s) in `.env` (single: `DATA_PATH`, multiple: `DATA_PATHS`)
3. Run full pipeline
4. Inspect saved outputs (`artifacts/metrics.json`)
5. Run API and test `/docs`
6. Run dashboard and manually check recommendation behavior

Example (Windows PowerShell):

```powershell
$env:DATA_PATHS="Gold Price Dynamics/GoldUSD.csv,data/XAU_1d_data.csv"
python -m src.pipeline
```

## How to Check Output

### A) Check Console Output

After running:

```bash
python -m src.pipeline
```

You should see:

- `Pipeline completed.`
- JSON metrics like `mae`, `rmse`, `r2`, `mape_pct`, `directional_accuracy_pct`

### B) Check Saved Files

Open the `artifacts/` folder and confirm these files exist:

- `xgb_model.joblib` (trained model)
- `x_scaler.joblib` and `y_scaler.joblib` (scalers)
- `xgb_predictions.joblib` (predicted values)
- `metrics.json` (evaluation summary)

### C) Check API Output

Start API:

```bash
uvicorn src.api:app --reload
```

Then open:

- Swagger docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

You should receive:

```json
{"status":"ok"}
```

### D) Check Dashboard Output

Start dashboard:

```bash
streamlit run dashboard/app.py
```

In browser (`http://localhost:8501`):

- enter current price, predicted price, interval low/high
- click **Get Recommendation**
- verify signal output (`BUY`, `HOLD`, `SELL`) and confidence field
- check **Automatic Daily Forecast** section for tomorrow trend and next-week price table

## Troubleshooting

- **`FileNotFoundError` for dataset**
  - Ensure `DATA_PATH` or each comma-separated path in `DATA_PATHS` points to a real CSV file.
- **Date parsing warnings**
  - Current loader handles known formats, but mixed custom formats may require adding one more explicit parse rule.
- **Module not found**
  - Run `pip install -r requirements.txt` in the same Python environment.
- **API/Dashboard port already in use**
  - Stop previous process or run on another port.

## Output Artifacts

After running the pipeline, outputs are stored in `artifacts/`:

- `xgb_model.joblib`
- `x_scaler.joblib`
- `y_scaler.joblib`
- `xgb_predictions.joblib`
- `metrics.json`

## Evaluation Metrics

Current pipeline reports:

- MAE
- RMSE
- R2
- MAPE
- Directional Accuracy

## Development Notes

- The notebook is retained for experimentation; `src/` is the primary production path.
- Raw datasets are intentionally not committed by default.
- Keep walk-forward validation practices to avoid optimistic test performance.

## Roadmap

- Add external macro/sentiment features (CPI, DXY, rates, event signals).
- Add confidence intervals and forecast bands for risk-aware decisions.
- Add model registry and scheduled retraining.
- Add CSV export for prediction review in spreadsheet tools.