"""Download the Kaggle dataset used by this project (already included as data/india_2000_2024_daily_weather.csv).
Dataset: https://www.kaggle.com/datasets/developerghost/climate-in-india-daily-weather-data-2000-2024
"""
import shutil, glob
from pathlib import Path
import kagglehub

path = kagglehub.dataset_download("developerghost/climate-in-india-daily-weather-data-2000-2024")
for f in glob.glob(f"{path}/*.csv"):
    shutil.copy(f, Path(__file__).parent)
    print("saved", Path(f).name)
