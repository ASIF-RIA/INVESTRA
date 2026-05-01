from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.forecasting import save_daily_forecast


if __name__ == "__main__":
    result = save_daily_forecast(days=7)
    print("Daily forecast updated.")
    print(result["tomorrow_prediction"])
