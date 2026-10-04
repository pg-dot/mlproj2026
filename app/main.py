"""
FastAPI backend serving the trained SVM-RBF heatwave early-warning model + the website.

    pip install -r requirements.txt
    uvicorn app.main:app --reload        # then open http://127.0.0.1:8000

Endpoints
  GET  /api/health
  GET  /api/model                 model card (hyper-parameters, #support vectors, threshold)
  POST /api/predict               manual observations -> tomorrow's heatwave probability
  GET  /api/live/{city}?day=YYYY-MM-DD   real weather for that day (default today) from Open-Meteo -> next-day forecast
"""
import sys
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
from pathlib import Path

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import (CITIES, FEATURES, THRESHOLD, MODEL_JSON, make_features, alert_level,  # noqa: E402
                       normal, imd_threshold)

model = joblib.load(ROOT / "models" / "svm_rbf.joblib")
svm = model[-1]
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
app = FastAPI(title="Heatwave Early Warning — SVM (RBF kernel)", version="1.0")


class Obs(BaseModel):
    city: str = Field(examples=["Delhi"])
    today: date = Field(examples=["2024-05-28"])
    tmax: float = Field(description="Today's max temperature °C", examples=[45.9])
    tmax_1: float = Field(description="Yesterday's max temperature °C", examples=[45.1])
    tmax_2: float = Field(description="Max temperature 2 days ago °C", examples=[44.0])
    tmin: float = Field(examples=[30.5])
    app_tmax: float = Field(description="Today's feels-like max °C", examples=[44.0])
    rain7: float = Field(0.0, description="Rainfall over last 7 days, mm")
    wind_max: float = Field(12.0, description="Today's max wind speed km/h")


def predict(o: Obs) -> dict:
    if o.city not in CITIES:
        raise HTTPException(400, f"city must be one of {list(CITIES)}")
    f = make_features(o.city, o.today, o.tmax, o.tmax_1, o.tmax_2, o.tmin, o.app_tmax, o.rain7, o.wind_max)
    x = np.array([[f[k] for k in FEATURES]])
    p = float(model.predict_proba(x)[0, 1])
    d = float(model.decision_function(x)[0])
    # explain: support vectors with the largest kernel contribution alpha_i*y_i*K(x, sv_i)
    z = model[0].transform(x)[0]
    k = np.exp(-svm._gamma * ((svm.support_vectors_ - z) ** 2).sum(1))
    contrib = svm.dual_coef_[0] * k
    top = np.argsort(-np.abs(contrib))[:5]
    return dict(city=o.city, forecast_for=str(o.today + timedelta(days=1)), probability=round(p, 4),
                decision_function=round(d, 4), threshold=THRESHOLD, heatwave_predicted=p >= THRESHOLD,
                **alert_level(p), features={k: round(v, 3) for k, v in f.items()},
                normal_tomorrow=round(normal(o.city, o.today + timedelta(days=1)), 1),
                imd_threshold=imd_threshold(o.city),
                top_support_vectors=[dict(index=int(i), contribution=round(float(contrib[i]), 4),
                                          similarity=round(float(k[i]), 4),
                                          cls="heatwave" if svm.dual_coef_[0][i] > 0 else "normal") for i in top])


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model")
def model_card():
    return dict(kernel="rbf", C=svm.C, gamma=float(svm._gamma), class_weight=str(svm.class_weight),
                n_support=svm.n_support_.tolist(), threshold=THRESHOLD, features=FEATURES,
                cities=list(CITIES))


@app.post("/api/predict")
def api_predict(o: Obs):
    return predict(o)


DAILY = "temperature_2m_max,temperature_2m_min,apparent_temperature_max,rain_sum,wind_speed_10m_max"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@app.get("/api/live/{city}")
async def live(city: str, day: date | None = None):
    """Fetch real weather for `day` (default: today, IST) from Open-Meteo and predict the next day.

    Open-Meteo serves the same variables the Kaggle dataset was built from. Recent days (last ~3 months,
    including today) come from the forecast API; older days come from the historical archive API.
    """
    if city not in CITIES:
        raise HTTPException(404, "unknown city")
    today_ist = datetime.now(IST).date()
    day = day or today_ist
    if day > today_ist:
        raise HTTPException(400, "Pick today or a past date. The model needs that day's observed weather.")
    c = CITIES[city]
    base = dict(latitude=c["lat"], longitude=c["lon"], timezone="Asia/Kolkata", daily=DAILY)
    if (today_ist - day).days <= 85:
        url, params = OPEN_METEO_URL, dict(base, past_days=92, forecast_days=1)
    else:
        url, params = ARCHIVE_URL, dict(base, start_date=str(day - timedelta(days=7)), end_date=str(day))
    try:
        async with httpx.AsyncClient(timeout=20) as cl:
            r = await cl.get(url, params=params)
        if r.status_code == 429:
            raise HTTPException(503, "Open-Meteo's free daily request limit is used up on this network. Try again tomorrow.")
        r.raise_for_status()
        d = r.json()["daily"]
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise HTTPException(502, f"Could not reach Open-Meteo ({type(e).__name__}). Check your internet connection.")
    if str(day) not in d["time"]:
        raise HTTPException(404, f"Open-Meteo has no data for {day} yet.")
    i = d["time"].index(str(day))
    need = [d["temperature_2m_max"][i - 2:i + 1], [d["temperature_2m_min"][i], d["apparent_temperature_max"][i],
                                                   d["wind_speed_10m_max"][i]]]
    if i < 6 or any(v is None for part in need for v in part):
        raise HTTPException(404, f"Open-Meteo has incomplete data around {day}. Try an earlier date.")
    o = Obs(city=city, today=day, tmax=d["temperature_2m_max"][i],
            tmax_1=d["temperature_2m_max"][i - 1], tmax_2=d["temperature_2m_max"][i - 2],
            tmin=d["temperature_2m_min"][i], app_tmax=d["apparent_temperature_max"][i],
            rain7=float(sum(v or 0 for v in d["rain_sum"][i - 6:i + 1])), wind_max=d["wind_speed_10m_max"][i])
    out = predict(o)
    out["observations"] = o.model_dump(mode="json")
    out["source"] = "Open-Meteo " + ("forecast" if url == OPEN_METEO_URL else "historical archive") + " API"
    return out


WEB = ROOT / "web"
app.mount("/data", StaticFiles(directory=WEB / "data"), name="data")
app.mount("/js", StaticFiles(directory=WEB / "js"), name="js")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
