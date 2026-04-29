import pandas as pd
from prophet import Prophet


def train_prophet(df: pd.DataFrame, train_size: int):
    df_p = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    train_df = df_p.iloc[:train_size]
    test_df = df_p.iloc[train_size:]

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(train_df)
    forecast = model.predict(test_df[["ds"]])
    return model, forecast["yhat"].values
