from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.data_ingestion import load_gold_data
from src.features import build_features, training_matrix
from src.models.train_xgb import train_xgb


@dataclass(frozen=True)
class ForecastPoint:
    forecast_date: str
    predicted_close: float


def _load_or_train_artifacts() -> tuple[object, object, object]:
    model_dir = Path(settings.model_dir)
    model_path = model_dir / "xgb_model.joblib"
    x_scaler_path = model_dir / "x_scaler.joblib"
    y_scaler_path = model_dir / "y_scaler.joblib"

    if not (model_path.exists() and x_scaler_path.exists() and y_scaler_path.exists()):
        train_xgb()

    model = joblib.load(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    return model, x_scaler, y_scaler


def _predict_next_close(model: object, x_scaler: object, y_scaler: object, feature_df: pd.DataFrame) -> float:
    x, _ = training_matrix(feature_df)
    latest_x = x.tail(1)
    latest_scaled = x_scaler.transform(latest_x)
    pred_scaled = model.predict(latest_scaled)
    pred = y_scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).flatten()[0]
    return float(pred)


def generate_multi_day_forecast(days: int = 7) -> dict:
    if days < 1:
        raise ValueError("days must be >= 1")

    model, x_scaler, y_scaler = _load_or_train_artifacts()
    history_df = load_gold_data(settings.data_paths)
    work_df = history_df.copy()

    last_history_date = work_df.index.max().date()
    start_date = max(date.today(), last_history_date)
    points: list[ForecastPoint] = []

    for offset in range(1, days + 1):
        feat_df = build_features(work_df)
        if feat_df.empty:
            raise ValueError("Not enough rows to build features. Add more historical data.")

        pred_close = _predict_next_close(model, x_scaler, y_scaler, feat_df)
        next_day = start_date + timedelta(days=offset)
        points.append(ForecastPoint(forecast_date=next_day.isoformat(), predicted_close=pred_close))

        synthetic_row = pd.DataFrame(
            {
                "Open": [pred_close],
                "High": [pred_close],
                "Low": [pred_close],
                "Close": [pred_close],
                "Volume": [float(work_df["Volume"].iloc[-1]) if "Volume" in work_df.columns else 0.0],
            },
            index=[pd.Timestamp(next_day)],
        )
        work_df = pd.concat([work_df, synthetic_row], axis=0)

    tomorrow = points[0]
    last_close = float(history_df["Close"].iloc[-1])
    direction = "increase" if tomorrow.predicted_close >= last_close else "decrease"

    return {
        "generated_on": date.today().isoformat(),
        "model": "XGBoost",
        "last_known_close": last_close,
        "tomorrow_prediction": {
            "date": tomorrow.forecast_date,
            "predicted_close": tomorrow.predicted_close,
            "direction_vs_today": direction,
        },
        "next_week_predictions": [point.__dict__ for point in points],
    }


def save_daily_forecast(output_path: str | Path | None = None, days: int = 7) -> dict:
    forecast = generate_multi_day_forecast(days=days)
    path = Path(output_path) if output_path else Path(settings.model_dir) / "daily_forecast.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(forecast, f, indent=2)
    return forecast
