from fastapi import FastAPI
from pydantic import BaseModel

from src.recommendation import recommend

app = FastAPI(title="INVESTRA Gold Forecast API", version="0.1.0")


class RecommendationInput(BaseModel):
    current_price: float
    predicted_price: float
    interval_low: float
    interval_high: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend_endpoint(payload: RecommendationInput):
    width_pct = (payload.interval_high - payload.interval_low) / payload.current_price
    return recommend(
        current_price=payload.current_price,
        predicted_price=payload.predicted_price,
        interval_width_pct=width_pct,
    )
