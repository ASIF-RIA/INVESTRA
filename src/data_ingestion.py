import pandas as pd


def _load_single_dataset(path: str) -> pd.DataFrame:
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


def _merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    if len(datasets) == 1:
        return datasets[0]

    merged = datasets[0].copy()
    for idx, extra_df in enumerate(datasets[1:], start=2):
        renamed = extra_df.rename(columns={col: f"{col}_ds{idx}" for col in extra_df.columns})
        merged = merged.join(renamed, how="inner")

    return merged.sort_index()


def load_gold_data(paths: str | tuple[str, ...] | list[str]) -> pd.DataFrame:
    if isinstance(paths, str):
        path_list = [paths]
    else:
        path_list = [path for path in paths if path]

    datasets = [_load_single_dataset(path) for path in path_list]
    return _merge_datasets(datasets)
