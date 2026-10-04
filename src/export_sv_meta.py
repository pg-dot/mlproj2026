"""Attach metadata (city, date, outcome) to each support vector so the website can say
which remembered training days drove a forecast. Run after train.py."""
import json, sys
from pathlib import Path
import joblib, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from features import load_raw, climatology, build_features, split, FEATURES

raw = load_raw(ROOT / "data" / "india_2000_2024_daily_weather.csv")
tr, _ = split(build_features(raw, climatology(raw)))
model = joblib.load(ROOT / "models" / "svm_rbf.joblib")
svm = model[-1]
rows = tr.iloc[svm.support_]
# sanity: stored support vectors equal the scaled training rows
assert np.allclose(model[0].transform(rows[FEATURES].values), svm.support_vectors_)
p = ROOT / "web" / "data" / "model.json"
m = json.loads(p.read_text())
m["sv_meta"] = [[c, d.strftime("%Y-%m-%d"), int(y), round(float(t), 1)]
                for c, d, y, t in zip(rows.city, rows.target_date, rows.y, rows.tmax_next)]
p.write_text(json.dumps(m, separators=(",", ":")))
print(len(m["sv_meta"]), m["sv_meta"][:2])
