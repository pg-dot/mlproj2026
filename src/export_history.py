"""Bundle the Kaggle daily data for the website so picking any date from 2000-2024 auto-fills
the inputs. Values are stored x10 as integers to keep the file small."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from features import load_raw

df = load_raw(ROOT / "data" / "india_2000_2024_daily_weather.csv")
out = {"start": str(df.date.min().date()), "end": str(df.date.max().date()), "cities": {}}
for city, g in df.groupby("city"):
    g = g.sort_values("date")
    assert len(g) == (g.date.max() - g.date.min()).days + 1  # no gaps
    out["cities"][city] = {k: [int(round(v * 10)) for v in g[c]] for k, c in
                           [("tmax", "tmax"), ("tmin", "tmin"), ("app", "app_tmax"), ("rain", "rain"), ("wind", "wind_max")]}
p = ROOT / "web" / "data" / "history.json"
p.write_text(json.dumps(out, separators=(",", ":")))
print(p, round(p.stat().st_size / 1e6, 2), "MB")
