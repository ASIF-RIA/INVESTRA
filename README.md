# INVESTRA

Gold price forecasting and investor decision-support toolkit.

## Project Structure

- `Gold_price_prediction.ipynb`: original research notebook
- `src/data_ingestion.py`: loads and cleans OHLC data
- `src/features.py`: feature engineering and training matrix creation
- `src/models/train_xgb.py`: leakage-safe XGBoost training pipeline
- `src/models/train_lstm.py`: baseline LSTM trainer
- `src/models/train_prophet.py`: baseline Prophet trainer
- `src/models/ensemble.py`: weighted ensemble helper
- `src/evaluate.py`: MAE/RMSE/MAPE/R2/directional accuracy utilities
- `src/recommendation.py`: Buy/Hold/Sell signal logic
- `src/api.py`: FastAPI service
- `dashboard/app.py`: Streamlit UI starter

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Configure environment:
   - copy `.env.example` to `.env` and edit values
3. Train XGBoost artifacts:
   - `python -m src.models.train_xgb`
4. Run full train + evaluate pipeline:
   - `python -m src.pipeline`
5. Run API:
   - `uvicorn src.api:app --reload`
6. Run dashboard:
   - `streamlit run dashboard/app.py`

## Notes

- The current notebook contains experimental work; production flow should use `src/` modules.
- Keep walk-forward validation and avoid using the test split for early stopping or hyperparameter tuning.