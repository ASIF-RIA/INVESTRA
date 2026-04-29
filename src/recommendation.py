def recommend(current_price: float, predicted_price: float, interval_width_pct: float) -> dict:
    roi = (predicted_price - current_price) / current_price

    if roi > 0.01 and interval_width_pct < 0.02:
        signal = "BUY"
    elif roi < -0.01 or interval_width_pct > 0.05:
        signal = "SELL"
    else:
        signal = "HOLD"

    if interval_width_pct < 0.02:
        confidence = "HIGH"
    elif interval_width_pct < 0.05:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "signal": signal,
        "confidence": confidence,
        "expected_roi_pct": round(roi * 100, 3),
        "risk_pct": round(interval_width_pct * 100, 3),
    }
