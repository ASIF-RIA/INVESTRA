from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Ensure root package imports work when running as a script.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.data_ingestion import load_gold_data
from src.evaluate import evaluate_all
from src.features import build_features, training_matrix
from src.models.train_lstm import train_lstm
from src.models.train_prophet import train_prophet
from src.models.train_xgb import train_xgb


def _align_to_min_length(*arrays: np.ndarray) -> list[np.ndarray]:
    n = min(len(arr) for arr in arrays)
    return [np.asarray(arr)[-n:] for arr in arrays]


def _save_metrics(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _classification_scores(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Convert regression outputs to direction classification.
    true_dir = (np.diff(y_true) > 0).astype(int)
    pred_dir = (np.diff(y_pred) > 0).astype(int)
    pred_score = np.diff(y_pred)

    metrics = {
        "precision": float(precision_score(true_dir, pred_dir, zero_division=0)),
        "recall": float(recall_score(true_dir, pred_dir, zero_division=0)),
        "f1_score": float(f1_score(true_dir, pred_dir, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_dir, pred_score)),
    }
    return metrics, true_dir, pred_score


def _plot_roc_curves(
    roc_inputs: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    for model_name, (true_dir, pred_score) in roc_inputs.items():
        fpr, tpr, _ = roc_curve(true_dir, pred_score)
        auc_val = roc_auc_score(true_dir, pred_score)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Baseline")
    plt.title("ROC Curve (Directional Movement Classification)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_prediction_comparison(
    y_true: np.ndarray,
    lr_pred: np.ndarray,
    xgb_pred: np.ndarray,
    lstm_pred: np.ndarray,
    prophet_pred: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label="Actual Close", linewidth=2.0)
    plt.plot(lr_pred, label="Linear Regression Prediction", alpha=0.85)
    plt.plot(xgb_pred, label="XGBoost Prediction", alpha=0.85)
    plt.plot(lstm_pred, label="LSTM Prediction", alpha=0.85)
    plt.plot(prophet_pred, label="Prophet Prediction", alpha=0.85)
    plt.title("Gold Price Forecast Comparison")
    plt.xlabel("Time Steps (Aligned Test Window)")
    plt.ylabel("Gold Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_metric_comparison(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(metrics.keys())
    metric_groups = ["mae", "rmse", "mape_pct", "directional_accuracy_pct", "r2"]
    plot_titles = ["MAE", "RMSE", "MAPE (%)", "Directional Accuracy (%)", "R2"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, (metric_key, title) in enumerate(zip(metric_groups, plot_titles)):
        values = [metrics[name][metric_key] for name in model_names]
        axes[i].bar(model_names, values)
        axes[i].set_title(title)
        axes[i].tick_params(axis="x", rotation=15)
        axes[i].grid(axis="y", alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_comparison() -> dict[str, dict[str, float]]:
    # Build shared train/test split from engineered features.
    df_raw = load_gold_data(settings.data_paths)
    df_feat = build_features(df_raw)
    x, y = training_matrix(df_feat)
    split = int(len(df_feat) * settings.train_ratio)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Linear Regression baseline.
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)

    # XGBoost baseline.
    xgb_model, x_scaler, y_scaler, x_test_s, _ = train_xgb()
    xgb_pred_s = xgb_model.predict(x_test_s)
    xgb_pred = y_scaler.inverse_transform(xgb_pred_s.reshape(-1, 1)).flatten()

    # LSTM model using the same scaled features/target pipeline.
    x_scaler_lstm = StandardScaler()
    y_scaler_lstm = StandardScaler()
    x_train_s = x_scaler_lstm.fit_transform(x_train)
    x_test_s_lstm = x_scaler_lstm.transform(x_test)
    y_train_s = y_scaler_lstm.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    lstm_model = train_lstm(x_train_s, y_train_s)
    x_test_lstm = x_test_s_lstm.reshape((x_test_s_lstm.shape[0], 1, x_test_s_lstm.shape[1]))
    lstm_pred_s = lstm_model.predict(x_test_lstm, verbose=0).ravel()
    lstm_pred = y_scaler_lstm.inverse_transform(lstm_pred_s.reshape(-1, 1)).flatten()

    # Prophet model from indexed raw frame.
    _, prophet_pred = train_prophet(df_raw, train_size=split)

    y_true, lr_pred, xgb_pred, lstm_pred, prophet_pred = _align_to_min_length(
        y_test.values, lr_pred, xgb_pred, lstm_pred, prophet_pred
    )

    metrics = {
        "LinearRegression": evaluate_all(y_true, lr_pred),
        "XGBoost": evaluate_all(y_true, xgb_pred),
        "LSTM": evaluate_all(y_true, lstm_pred),
        "Prophet": evaluate_all(y_true, prophet_pred),
    }
    cls_metrics: dict[str, dict[str, float]] = {}
    roc_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, pred in [
        ("LinearRegression", lr_pred),
        ("XGBoost", xgb_pred),
        ("LSTM", lstm_pred),
        ("Prophet", prophet_pred),
    ]:
        model_cls_metrics, true_dir, pred_score = _classification_scores(y_true, pred)
        cls_metrics[name] = model_cls_metrics
        roc_inputs[name] = (true_dir, pred_score)

    figures_dir = Path("reports") / "figures"
    _plot_prediction_comparison(
        y_true=y_true,
        lr_pred=lr_pred,
        xgb_pred=xgb_pred,
        lstm_pred=lstm_pred,
        prophet_pred=prophet_pred,
        output_path=figures_dir / "prediction_comparison.png",
    )
    _plot_metric_comparison(metrics, output_path=figures_dir / "performance_comparison.png")
    _plot_roc_curves(roc_inputs, output_path=figures_dir / "roc_comparison.png")
    _save_metrics(metrics, Path("reports") / "comparison_metrics.json")
    _save_metrics(cls_metrics, Path("reports") / "classification_metrics.json")
    return metrics


if __name__ == "__main__":
    results = run_comparison()
    print("Comparison complete. Outputs saved under reports/.")
    print(json.dumps(results, indent=2))
