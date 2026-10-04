"""
Inference helpers shared by the API (app/main.py). The website re-implements the exact same
maths in JavaScript (web/js/svm.js) so the trained model can also run fully in the browser.

Prediction for tomorrow (t+1) from observations of today (t), yesterday (t-1), and t-2.
"""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_JSON = json.loads((ROOT / "web" / "data" / "model.json").read_text())
NORMALS = {c: np.array(v) for c, v in MODEL_JSON["normals"].items()}
CITIES = MODEL_JSON["cities"]
FEATURES = MODEL_JSON["features"]
THRESHOLD = MODEL_JSON["threshold"]


def _doy(d: date) -> int:
    return min(d.timetuple().tm_yday, 365)


def normal(city: str, d: date) -> float:
    return float(NORMALS[city][_doy(d) - 1])


def imd_threshold(city: str) -> float:
    return 37.0 if CITIES[city]["coastal"] else 40.0


def is_heatwave(city, tmax, dep):
    thr = imd_threshold(city)
    hw = (tmax >= thr and dep >= 4.5) or tmax >= 45.0
    severe = hw and (dep > 6.4 or tmax >= 47.0)
    return int(hw), int(severe)


def make_features(city: str, today: date, tmax: float, tmax_1: float, tmax_2: float, tmin: float,
                  app_tmax: float, rain7: float, wind_max: float) -> dict:
    """Build the 12 model features exactly as in training (src/features.py)."""
    thr = imd_threshold(city)
    dep = tmax - normal(city, today)
    dep1 = tmax_1 - normal(city, today - timedelta(days=1))
    dep2 = tmax_2 - normal(city, today - timedelta(days=2))
    dep_mean3 = (dep + dep1 + dep2) / 3.0
    normal_next = normal(city, today + timedelta(days=1))

    def margin(a):
        proj = normal_next + a
        return max(min(proj - thr, a - 4.5), proj - 45.0)

    f = dict(margin=margin(dep), margin3=margin(dep_mean3), dep=dep, dep_mean3=dep_mean3,
             tmax_trend3=tmax - tmax_2, app_tmax=app_tmax, dtr=tmax - tmin,
             rain7_log=float(np.log1p(max(rain7, 0.0))), wind_max=wind_max, normal_next=normal_next,
             hw_today=is_heatwave(city, tmax, dep)[0], coastal=int(CITIES[city]["coastal"]))
    return f


def alert_level(p: float) -> dict:
    """IMD-style colour-coded alert from the calibrated probability."""
    t = THRESHOLD
    if p >= max(0.5, 3 * t):
        return dict(level="Red", action="Take action — heatwave very likely tomorrow")
    if p >= t:
        return dict(level="Orange", action="Be prepared — heatwave expected tomorrow")
    if p >= t / 2:
        return dict(level="Yellow", action="Be updated — elevated heat, watch conditions")
    return dict(level="Green", action="No action — heatwave unlikely")
