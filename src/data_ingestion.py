import pandas as pd


def load_gold_data(path: str) -> pd.DataFrame:
    # Try default CSV first, then fallback to semicolon-delimited files.
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        df = pd.read_csv(path, sep=";")

    # Parse known vendor formats first to avoid parser warnings.
    date_series = pd.to_datetime(df["Date"], format="%d-%m-%y", errors="coerce")
    missing_mask = date_series.isna()
    if missing_mask.any():
        date_series.loc[missing_mask] = pd.to_datetime(
            df.loc[missing_mask, "Date"], format="%Y.%m.%d %H:%M", errors="coerce"
        )
    df["Date"] = date_series
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates().dropna(subset=["Date", *cols])
    df = df.set_index("Date").sort_index()
    return df
