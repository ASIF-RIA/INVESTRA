import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA_7"] = out["Close"].rolling(7).mean()
    out["MA_30"] = out["Close"].rolling(30).mean()
    out["Lag_1"] = out["Close"].shift(1)
    out["Lag_2"] = out["Close"].shift(2)
    out["Lag_3"] = out["Close"].shift(3)
    out["Volatility"] = out["High"] - out["Low"]
    out["Return_1"] = out["Close"].pct_change(1)
    out["Return_7"] = out["Close"].pct_change(7)
    out = out.dropna()
    return out


def training_matrix(df: pd.DataFrame):
    features = ["Lag_1", "Lag_2", "Lag_3", "MA_7", "MA_30", "Volatility", "Return_1", "Return_7"]
    x = df[features]
    y = df["Close"]
    return x, y
