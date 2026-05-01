from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_lower_better(values: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(values.values()), dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return {k: 1.0 for k in values}
    return {k: float((vmax - v) / (vmax - vmin)) for k, v in values.items()}


def _normalize_higher_better(values: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(values.values()), dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return {k: 1.0 for k in values}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}


def build_decision_plot() -> None:
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    reg = _load_json(reports_dir / "comparison_metrics.json")
    cls = _load_json(reports_dir / "classification_metrics.json")

    # Exclude likely-leaky model from practical ranking chart.
    candidate_models = [m for m in reg.keys() if m != "LinearRegression"]

    mae = {m: reg[m]["mae"] for m in candidate_models}
    rmse = {m: reg[m]["rmse"] for m in candidate_models}
    mape = {m: reg[m]["mape_pct"] for m in candidate_models}
    da = {m: reg[m]["directional_accuracy_pct"] for m in candidate_models}
    f1 = {m: cls[m]["f1_score"] for m in candidate_models}
    auc = {m: cls[m]["roc_auc"] for m in candidate_models}

    score_components = [
        _normalize_lower_better(mae),
        _normalize_lower_better(rmse),
        _normalize_lower_better(mape),
        _normalize_higher_better(da),
        _normalize_higher_better(f1),
        _normalize_higher_better(auc),
    ]

    final_scores = {}
    for m in candidate_models:
        final_scores[m] = float(np.mean([comp[m] for comp in score_components]))

    ranked_models = sorted(candidate_models, key=lambda x: final_scores[x], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: key metric bars.
    x = np.arange(len(candidate_models))
    width = 0.25
    axes[0].bar(x - width, [da[m] for m in candidate_models], width, label="Directional Acc. (%)")
    axes[0].bar(x, [f1[m] * 100 for m in candidate_models], width, label="F1-score (%)")
    axes[0].bar(x + width, [auc[m] * 100 for m in candidate_models], width, label="ROC-AUC (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(candidate_models, rotation=15)
    axes[0].set_ylabel("Percentage")
    axes[0].set_title("Decision-Critical Metric Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Right: composite ranking.
    axes[1].barh(ranked_models[::-1], [final_scores[m] for m in ranked_models[::-1]])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Composite Score (0-1)")
    axes[1].set_title("Final Model Ranking (Leakage-Filtered)")
    axes[1].grid(axis="x", alpha=0.3)

    best_model = ranked_models[0]
    fig.suptitle(f"Final Decision Visualization - Best Model: {best_model}", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "final_decision_graph.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_decision_plot()
    print("Saved reports/figures/final_decision_graph.png")
