from pathlib import Path
import json

import joblib

from src.config import settings
from src.data_ingestion import load_gold_data
from src.features import build_features, training_matrix
from src.evaluate import evaluate_all
from src.models.train_xgb import train_xgb


def run_pipeline() -> dict:
    df = load_gold_data(settings.data_paths)
    df_feat = build_features(df)
    x, y = training_matrix(df_feat)
    split = int(len(df_feat) * settings.train_ratio)
    y_test = y.iloc[split:]

    model, x_scaler, y_scaler, x_test_scaled, _ = train_xgb()
    pred_scaled = model.predict(x_test_scaled)
    predictions = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    metrics = evaluate_all(y_test.values, predictions)

    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictions, model_dir / "xgb_predictions.joblib")
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline completed.")
    print(json.dumps(result, indent=2))
