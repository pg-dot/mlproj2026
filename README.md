# Heatline: Heatwave Early Warning using Kernel-Based Classification

**Technique:** Support Vector Machine (SVM) with an RBF kernel
**Use case:** KJS-CES-01, Climate Intelligence for Heatwave Monitoring, Prediction and Early Warning (IMD Mumbai–Pune)

The name refers to the line the SVM draws between heatwave and normal days.

On the evening of day *t*, the system predicts whether day *t+1* will be a heatwave day under the India Meteorological Department (IMD) criteria. It covers 10 Indian cities. It then issues an IMD-style Green / Yellow / Orange / Red alert and explains which remembered training days (support vectors) drove the decision.

## Results on unseen years (2021–2024, 6,120 city-days, 194 heatwave days)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Onset recall |
|---|---|---|---|---|---|---|
| **SVM, RBF kernel** | 0.650 | 0.972 | **0.665** | 0.637 | **0.696** | 0.145 |
| Logistic regression | 0.665 | 0.976 | 0.646 | 0.638 | 0.655 | 0.145 |
| Linear SVM | 0.660 | 0.976 | 0.654 | 0.620 | 0.691 | 0.145 |
| Persistence (today = tomorrow) | 0.426 | 0.816 | 0.644 | 0.644 | 0.644 | 0.000 |

- The SVM caught 135 of 194 heatwave days a day ahead, with 77 false alarms in 5,926 normal days.
- It caught **88.9% of severe heatwave days**.
- Best hyper-parameters: C = 10, γ = 0.01, heatwave class weight 5:1, giving 1,446 support vectors. The warning threshold is P ≥ 0.105.
- The linear models rank days almost as well. The engineered *IMD margin* features already linearise much of the problem, so the kernel's gain is mainly a better operating point (higher F1 and recall).

## Project structure

```
data/      india_2000_2024_daily_weather.csv (Kaggle) + download_data.py
src/       features.py   IMD labels + 12 engineered features (no leakage)
           train.py      tuning, training, baselines, evaluation, exports
           inference.py  feature builder + alert levels used by the API
           export_sv_meta.py  city/date of each support vector (for explanations)
models/    svm_rbf.joblib (StandardScaler -> SVC rbf), metrics.json, verification.json
app/       main.py       FastAPI backend: /api/predict, /api/live/{city}, serves the website
web/       index.html    the website;  js/svm.js  exact JS re-implementation of the SVM
           data/         model.json (support vectors, α, b, Platt A/B), results.json, kernel_demo.json
notebooks/ heatwave_svm_rbf.ipynb  executed walk-through (EDA → labels → model → evaluation)
```

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # open http://127.0.0.1:8000
# optional
python src/train.py                    # retrain everything (~7 min on 2 cores)
python notebooks/build_notebook.py     # regenerate the notebook
```

The website also works as plain static files (`cd web && python -m http.server`). The trained SVM runs in the browser using the exported support vectors and gives the same probabilities as Python (checked to 1e-4 against `models/verification.json`). Picking a city and date fills the inputs automatically. Dates from 2000–2024 come from the bundled Kaggle data (`web/data/history.json`, built by `src/export_history.py`). Later dates, including today, are fetched from Open-Meteo by the FastAPI server, so they only work when you run it locally.

### API example
```bash
curl -X POST localhost:8000/api/predict -H "content-type: application/json" -d '{
  "city":"Delhi","today":"2024-05-28","tmax":45.9,"tmax_1":45.1,"tmax_2":44.0,
  "tmin":30.5,"app_tmax":44.0,"rain7":0,"wind_max":12}'
curl localhost:8000/api/live/Delhi                     # today's weather -> tomorrow's outlook
curl "localhost:8000/api/live/Delhi?day=2025-05-20"    # any past date
```
Interactive API docs: http://127.0.0.1:8000/docs

## Method

1. **Data:** 91,320 daily rows covering 10 cities from 2000 to 2024. Variables are Tmax, Tmin, feels-like temperature, rain and wind. There are no missing values.
2. **Labels (IMD criteria):**
   - *Normal* is the per-city, day-of-year mean Tmax over 2000–2020 only, smoothed over 31 days.
   - A day is a **heatwave** if Tmax ≥ 40 °C (37 °C for the coastal cities Mumbai, Kolkata and Chennai) **and** the departure from normal is ≥ 4.5 °C, or if Tmax ≥ 45 °C.
   - A heatwave is **severe** if the departure is > 6.4 °C or Tmax ≥ 47 °C.
   - This gives 625 heatwave days, 87 of them severe.
3. **Early-warning framing:** features use data up to day *t*; the target is day *t+1*. Only target days from March to July are kept (the heatwave season).
4. **Features (12):**
   - IMD margin from today's anomaly, and from the 3-day anomaly
   - departure today and the 3-day mean departure
   - 3-day Tmax trend
   - feels-like Tmax
   - diurnal range
   - log 7-day rain
   - max wind
   - tomorrow's normal
   - heatwave-today flag
   - coastal flag
5. **Model:**
   - `StandardScaler` → `SVC(kernel="rbf", probability=True)`.
   - C, γ and the class weight are tuned with **year-blocked 5-fold CV** (`GroupKFold` by year). Scoring is PR-AUC, because positives are only ~1.3% of rows.
   - Platt scaling turns the output into probabilities. The threshold is the one that maximises out-of-fold F1.
6. **Test:** 2021–2024 is scored once, against logistic regression, a linear SVM and persistence.
7. **Explainability:** permutation importance, plus the top support-vector votes α·y·K(x, sv) for each forecast.

## Limitations
- The data is Open-Meteo reanalysis. On extreme days it reads ~1–2 °C cooler than IMD station thermometers, so event counts are conservative.
- Labels are per city. IMD's official declaration also needs ≥ 2 stations in a sub-division on 2 consecutive days.
- Only observed weather is used. The first day of a spell (onset) is hard to call, at 14.5% recall. Adding NWP forecast Tmax would help most.

## Sources
- Dataset: https://www.kaggle.com/datasets/developerghost/climate-in-india-daily-weather-data-2000-2024
- IMD heatwave criteria (FAQ): https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf
- Open-Meteo API (live data; the dataset's origin): https://open-meteo.com/en/docs
- scikit-learn SVM guide: https://scikit-learn.org/stable/modules/svm.html
- scikit-learn SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- LIBSVM / Platt scaling: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
