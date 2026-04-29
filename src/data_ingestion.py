import pandas as pd


def load_gold_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df = df.set_index("Date").sort_index()

    cols = ["Open", "High", "Low", "Close", "Volume"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates().dropna(subset=cols)
    return df
