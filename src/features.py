"""
Feature engineering and IMD-based heatwave labelling.

Label definition follows the India Meteorological Department (IMD) heat-wave criteria
(https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf):

  Plains   : Tmax >= 40 C  AND departure from normal >= 4.5 C
  Coastal  : Tmax >= 37 C  AND departure from normal >= 4.5 C
  Any      : Tmax >= 45 C  (actual-maximum criterion)
  Severe   : heatwave AND (departure > 6.4 C OR Tmax >= 47 C)

"Normal" = per-city, day-of-year climatological mean Tmax, computed ONLY from the
development years (2000-2020) and smoothed with a 31-day circular moving average, so no
information from validation/test years leaks into the labels or features.

Early-warning framing: every row describes the weather observed up to day t and the
target is whether day t+1 will be a heatwave day (24-hour lead time).
"""
import numpy as np
import pandas as pd

CITIES = {
    # name: (lat, lon, coastal)
    "Delhi": (28.61, 77.21, 0),
    "Mumbai": (19.08, 72.88, 1),
    "Kolkata": (22.57, 88.36, 1),
    "Chennai": (13.08, 80.27, 1),
    "Bangalore": (12.97, 77.59, 0),
    "Hyderabad": (17.39, 78.49, 0),
    "Ahmedabad": (23.02, 72.57, 0),
    "Pune": (18.52, 73.86, 0),
    "Jaipur": (26.91, 75.79, 0),
    "Lucknow": (26.85, 80.95, 0),
}
COASTAL = {c for c, v in CITIES.items() if v[2]}
TRAIN_END = 2020                         # develop on 2000-2020, hold-out test 2021-2024
SEASON = (3, 7)                          # March-July: IMD heatwave season

FEATURES = [
    "margin", "margin3", "dep", "dep_mean3", "tmax_trend3", "app_tmax", "dtr",
    "rain7_log", "wind_max", "normal_next", "hw_today", "coastal",
]

FEATURE_LABELS = {
    "margin": "IMD margin, persisted anomaly (°C)",
    "margin3": "IMD margin, 3-day anomaly (°C)",
    "tmax": "Max temperature today (°C)",
    "tmin": "Min temperature today (°C)",
    "app_tmax": "Feels-like max today (°C)",
    "app_tmin": "Feels-like min today (°C)",
    "dtr": "Diurnal range Tmax−Tmin (°C)",
    "dep": "Departure from normal today (°C)",
    "dep_mean3": "3-day mean departure (°C)",
    "tmax_trend3": "3-day Tmax trend (°C)",
    "thr_gap": "Gap to IMD threshold (°C)",
    "hw_today": "Heatwave today (0/1)",
    "rain7_log": "7-day rainfall log(1+mm)",
    "wind_max": "Max wind speed (km/h)",
    "gust_max": "Max wind gust (km/h)",
    "wdir_sin": "Wind direction (sin)",
    "wdir_cos": "Wind direction (cos)",
    "normal_next": "Normal Tmax for tomorrow (°C)",
    "doy_sin": "Season (sin day-of-year)",
    "doy_cos": "Season (cos day-of-year)",
    "coastal": "Coastal city (0/1)",
}


def load_raw(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={
        "temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
        "apparent_temperature_max": "app_tmax", "apparent_temperature_min": "app_tmin",
        "precipitation_sum": "precip", "rain_sum": "rain",
        "wind_speed_10m_max": "wind_max", "wind_gusts_10m_max": "gust_max",
        "wind_direction_10m_dominant": "wdir",
    })
    return df.sort_values(["city", "date"]).reset_index(drop=True)


def doy(dates):
    return np.minimum(np.asarray(pd.DatetimeIndex(dates).dayofyear), 365)


def climatology(df, last_year=TRAIN_END, window=31):
    """Per-city smoothed day-of-year normal Tmax (365 values) from base years only."""
    base = df[df.date.dt.year <= last_year].copy()
    base["doy"] = doy(base.date)
    clim = base.groupby(["city", "doy"]).tmax.mean().unstack()
    half = window // 2
    out = {}
    for city, row in clim.iterrows():
        v = row.values
        padded = np.r_[v[-half:], v, v[:half]]
        out[city] = np.convolve(padded, np.ones(window) / window, "valid")
    return out  # dict city -> np.array(365)


def threshold(city):
    return 37.0 if city in COASTAL else 40.0


def label_heatwave(tmax, dep, coastal):
    thr = np.where(coastal == 1, 37.0, 40.0)
    hw = ((tmax >= thr) & (dep >= 4.5)) | (tmax >= 45.0)
    severe = hw & ((dep > 6.4) | (tmax >= 47.0))
    return hw.astype(int), severe.astype(int)


def build_features(df, normals):
    """Add labels + engineered features. Target = heatwave on day t+1."""
    parts = []
    for city, g in df.groupby("city", sort=False):
        g = g.sort_values("date").copy()
        d = doy(g.date)
        nm = normals[city]
        g["normal"] = nm[d - 1]
        g["coastal"] = int(city in COASTAL)
        g["dep"] = g.tmax - g.normal
        g["hw_today"], g["severe_today"] = label_heatwave(g.tmax.values, g.dep.values, g.coastal.values)
        g["dtr"] = g.tmax - g.tmin
        g["dep_mean3"] = g.dep.rolling(3, min_periods=1).mean()
        g["tmax_trend3"] = g.tmax - g.tmax.shift(2)
        g["thr_gap"] = g.tmax - threshold(city)
        g["rain7_log"] = np.log1p(g.rain.rolling(7, min_periods=1).sum())
        g["wdir_sin"] = np.sin(np.deg2rad(g.wdir))
        g["wdir_cos"] = np.cos(np.deg2rad(g.wdir))
        next_date = g.date + pd.Timedelta(days=1)
        nd = doy(next_date)
        g["target_date"] = next_date
        g["normal_next"] = nm[nd - 1]
        thr = threshold(city)
        # "IMD margin": how far tomorrow would be from meeting the IMD criteria if today's
        # anomaly persisted. >0 means the criteria would be met. A continuous version of the rule.
        for name, a in (("margin", g.dep), ("margin3", g.dep_mean3)):
            proj = g.normal_next + a
            g[name] = np.maximum(np.minimum(proj - thr, a - 4.5), proj - 45.0)
        g["doy_sin"] = np.sin(2 * np.pi * nd / 365.0)
        g["doy_cos"] = np.cos(2 * np.pi * nd / 365.0)
        # targets (tomorrow)
        g["y"] = g.hw_today.shift(-1)
        g["y_severe"] = g.severe_today.shift(-1)
        g["tmax_next"] = g.tmax.shift(-1)
        g["dep_next"] = g.dep.shift(-1)
        parts.append(g)
    out = pd.concat(parts).dropna(subset=["y", "tmax_trend3"])
    out["y"] = out.y.astype(int)
    out["y_severe"] = out.y_severe.astype(int)
    out["target_year"] = out.target_date.dt.year
    out["target_month"] = out.target_date.dt.month
    return out.reset_index(drop=True)


def in_season(frame):
    return frame.target_month.between(*SEASON)


def split(frame):
    s = frame[in_season(frame)]
    return s[s.target_year <= TRAIN_END], s[s.target_year > TRAIN_END]
