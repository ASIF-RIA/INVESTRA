from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.data_ingestion import load_gold_data
from src.features import build_features, training_matrix


def train_xgb():
    df = load_gold_data(settings.data_paths)
    df = build_features(df)
    x, y = training_matrix(df)

    split = int(len(df) * settings.train_ratio)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    y_train_s = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    val_size = max(1, int(len(x_train_s) * 0.1))
    x_tr, x_val = x_train_s[:-val_size], x_train_s[-val_size:]
    y_tr, y_val = y_train_s[:-val_size], y_train_s[-val_size:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=settings.random_state,
        early_stopping_rounds=10,
        eval_metric="rmse",
    )
    model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)

    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "xgb_model.joblib")
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, model_dir / "y_scaler.joblib")
    return model, x_scaler, y_scaler, x_test_s, y_test


if __name__ == "__main__":
    train_xgb()
    print("XGBoost artifacts saved.")
