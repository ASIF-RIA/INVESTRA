from pathlib import Path
import threading
import time

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.data_ingestion import load_gold_data
from src.features import build_features, training_matrix
from src.forecasting import save_daily_forecast
from src.recommendation import recommend

app = FastAPI(title="INVESTRA Gold Forecast API", version="0.1.0")


class RecommendationInput(BaseModel):
    current_price: float = Field(gt=0)
    predicted_price: float
    interval_low: float
    interval_high: float


class PredictResponse(BaseModel):
    predicted_price: float
    model: str
    source: str


class ForecastResponse(BaseModel):
    generated_on: str
    model: str
    last_known_close: float
    tomorrow_prediction: dict
    next_week_predictions: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", response_model=PredictResponse)
def predict_latest():
    model_dir = Path(settings.model_dir)
    model_path = model_dir / "xgb_model.joblib"
    x_scaler_path = model_dir / "x_scaler.joblib"
    y_scaler_path = model_dir / "y_scaler.joblib"
    required = [model_path, x_scaler_path, y_scaler_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing model artifacts. Run training first. Missing: {missing}",
        )

    df = load_gold_data(settings.data_paths)
    df_feat = build_features(df)
    x, _ = training_matrix(df_feat)
    if x.empty:
        raise HTTPException(status_code=400, detail="Not enough rows to build features for prediction.")

    latest_x = x.tail(1)
    model = joblib.load(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    latest_scaled = x_scaler.transform(latest_x)
    pred_scaled = model.predict(latest_scaled)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    return PredictResponse(
        predicted_price=float(pred),
        model="XGBoost",
        source="latest_available_features",
    )


@app.get("/forecast/latest", response_model=ForecastResponse)
def forecast_latest():
    try:
        forecast = save_daily_forecast(output_path=settings.forecast_file, days=7)
        return ForecastResponse(**forecast)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {exc}") from exc


@app.post("/recommend")
def recommend_endpoint(payload: RecommendationInput):
    width_pct = (payload.interval_high - payload.interval_low) / payload.current_price
    return recommend(
        current_price=payload.current_price,
        predicted_price=payload.predicted_price,
        interval_width_pct=width_pct,
    )


def _auto_refresh_forecast_loop() -> None:
    interval_hours = max(1, int(settings.forecast_refresh_hours))
    while True:
        try:
            save_daily_forecast(output_path=settings.forecast_file, days=7)
        except Exception:
            # Keep the service alive even if one refresh cycle fails.
            pass
        time.sleep(interval_hours * 3600)


@app.on_event("startup")
def startup_forecast_automation() -> None:
    # Produce initial forecast at startup and keep daily refresh in background.
    try:
        save_daily_forecast(output_path=settings.forecast_file, days=7)
    except Exception:
        pass
    t = threading.Thread(target=_auto_refresh_forecast_loop, daemon=True)
    t.start()
