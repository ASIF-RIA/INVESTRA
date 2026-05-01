from dataclasses import dataclass
import os


def _parse_data_paths() -> tuple[str, ...]:
    raw_paths = os.getenv("DATA_PATHS", "").strip()
    if raw_paths:
        return tuple(path.strip() for path in raw_paths.split(",") if path.strip())
    return (os.getenv("DATA_PATH", "./data/XAU_1d_data.csv"),)


@dataclass(frozen=True)
class Settings:
    data_paths: tuple[str, ...] = _parse_data_paths()
    model_dir: str = os.getenv("MODEL_DIR", "./artifacts")
    forecast_file: str = os.getenv("FORECAST_FILE", "./artifacts/daily_forecast.json")
    forecast_refresh_hours: int = int(os.getenv("FORECAST_REFRESH_HOURS", "24"))
    train_ratio: float = 0.8
    random_state: int = 42


settings = Settings()
