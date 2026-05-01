# Final Model Decision

## Objective
Select the best model for INVESTRA based on current evaluation outputs:
- Regression metrics (`MAE`, `RMSE`, `R2`, `MAPE`, `Directional Accuracy`)
- Directional classification metrics (`Precision`, `Recall`, `F1-score`, `ROC-AUC`)

Sources used:
- `reports/comparison_metrics.json`
- `reports/classification_metrics.json`

## Short Verdict
**Recommended best practical model: XGBoost**

## Why XGBoost is selected
1. It provides the **strongest directional trading behavior** among non-leakage candidates:
   - Directional Accuracy: `77.22%`
   - F1-score: `0.790`
   - ROC-AUC: `0.852`
2. INVESTRA is a **decision-support** system (`Buy/Hold/Sell`), so directional consistency is more critical than only minimizing point error.
3. It is stable and deployment-friendly for tabular engineered features.

## Important Note on Linear Regression
Linear Regression shows near-perfect scores (`R2 = 1.0`, `F1 = 1.0`, `AUC = 1.0`).  
This strongly indicates **data leakage / target leakage** in current feature setup, so it is **not accepted as the final production choice** until leakage-safe validation is enforced.

## Model-by-Model Summary
- **Linear Regression:** highest numeric scores, but likely invalid due to leakage.
- **XGBoost:** best reliable directional performance; best fit for recommendation engine.
- **Prophet:** best point-error metrics (`MAE/RMSE/MAPE`) but weak directional classification (`AUC ~ 0.492`).
- **LSTM:** currently underperforms vs XGBoost and Prophet in this setup.

## Final Decision
**Deploy XGBoost as the primary model** for current INVESTRA version.

## Optional Deployment Strategy
- Primary signal model: **XGBoost**
- Reference/monitoring model: **Prophet** (track point-error drift)
- Future improvement: fix leakage, rerun all comparisons, then re-validate final choice
