from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    data_path: str = os.getenv("DATA_PATH", "./data/XAU_1d_data.csv")
    model_dir: str = os.getenv("MODEL_DIR", "./artifacts")
    train_ratio: float = 0.8
    random_state: int = 42


settings = Settings()
