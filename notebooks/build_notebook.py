"""Generates and executes notebooks/heatwave_svm_rbf.ipynb (walk-through for viva / report)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Heatwave Early Warning using Kernel-Based Classification
**Technique:** Support Vector Machine with RBF kernel · **Use case:** KJS-CES-01 (IMD Mumbai–Pune)

Goal: on the evening of day *t*, predict whether day *t+1* will be a heatwave day under the IMD criteria, for 10 Indian cities.

Dataset: [India Daily Weather (2000–2024) – Major Cities, Kaggle](https://www.kaggle.com/datasets/developerghost/climate-in-india-daily-weather-data-2000-2024)"""),
code("""import sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '../src')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from features import load_raw, climatology, build_features, split, FEATURES, FEATURE_LABELS, CITIES
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
raw = load_raw('../data/india_2000_2024_daily_weather.csv')
print(raw.shape); raw.head()"""),
md("## 1. Exploratory analysis"),
code("""print(raw.isna().sum().sum(), 'missing values')
raw.groupby('city')[['tmax','tmin','rain','wind_max']].describe().round(1).iloc[:, [1,2,7,9,10,15]]"""),
code("""fig, ax = plt.subplots(figsize=(10,4))
for c in ['Delhi','Lucknow','Kolkata','Mumbai','Bangalore']:
    s = raw[raw.city==c].groupby(raw.date.dt.dayofyear).tmax.mean()
    ax.plot(s.index, s.values, label=c)
ax.axhline(40, ls='--', c='grey'); ax.text(5, 40.4, 'IMD plains threshold 40 °C', color='grey')
ax.set(xlabel='Day of year', ylabel='Mean Tmax (°C)', title='Seasonal cycle of maximum temperature'); ax.legend(); plt.show()"""),
md("""## 2. IMD-based labels
Normal = per-city day-of-year mean Tmax over 2000–2020, 31-day smoothed. Heatwave if Tmax ≥ 40 °C (37 °C coastal) **and** departure ≥ 4.5 °C, or Tmax ≥ 45 °C ([IMD FAQ](https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf))."""),
code("""normals = climatology(raw)
df = build_features(raw, normals)
print('heatwave days:', df.hw_today.sum(), '| severe:', df.severe_today.sum())
fig, axs = plt.subplots(1, 2, figsize=(12,3.6))
df.groupby(df.date.dt.year).hw_today.sum().plot.bar(ax=axs[0], color='#C8432B', title='Heatwave city-days per year')
df.groupby('city').hw_today.sum().sort_values().plot.barh(ax=axs[1], color='#C8432B', title='Heatwave days per city (2000–2024)')
plt.tight_layout(); plt.show()"""),
md("## 3. Features and temporal split"),
code("""tr, te = split(df)
print(f'development 2000–2020: {len(tr)} rows, {tr.y.sum()} heatwave ({tr.y.mean():.2%})')
print(f'test 2021–2024:        {len(te)} rows, {te.y.sum()} heatwave ({te.y.mean():.2%})')
pd.DataFrame({'feature': FEATURES, 'meaning': [FEATURE_LABELS[f] for f in FEATURES]})"""),
code("""fig, ax = plt.subplots(figsize=(6,5))
s = tr[tr.target_month.between(4,6)].sample(4000, random_state=0)
ax.scatter(s.dep, s.thr_gap, c=np.where(s.y==1,'#C8432B','#2F6DB5'), s=np.where(s.y==1,14,4), alpha=.6)
ax.set(xlabel="Today's departure from normal (°C)", ylabel="Today's Tmax − IMD threshold (°C)",
       title='Classes are not linearly separable → motivation for the RBF kernel'); plt.show()"""),
md("""## 4. SVM with RBF kernel
$K(x,x') = \\exp(-\\gamma\\lVert x-x'\\rVert^2)$, decision $f(x)=\\sum_i \\alpha_i y_i K(x_i,x)+b$.

`src/train.py` tunes **C**, **γ** and the heatwave class weight with year-blocked 5-fold CV (scoring = PR-AUC), chooses the warning threshold from out-of-fold probabilities, then trains on all of 2000–2020. Here we load the saved model and its tuning results."""),
code("""M = json.load(open('../models/metrics.json'))
print('best:', M['best'])
g = pd.DataFrame(M['grid']['pr_auc'], index=[f'C={c}' for c in M['grid']['C']], columns=[f'γ={x}' for x in M['grid']['gamma']])
g.style.background_gradient(cmap='Reds').format('{:.3f}')"""),
md("## 5. Hold-out evaluation (2021–2024)"),
code("""model = joblib.load('../models/svm_rbf.joblib')
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
p = model.predict_proba(te[FEATURES].values)[:,1]; thr = M['best']['threshold']
print(classification_report(te.y, (p>=thr).astype(int), target_names=['normal','heatwave'], digits=3))
pd.DataFrame(M['comparison']).set_index('model')[['pr_auc','roc_auc','f1','precision','recall','onset_recall','fp']]"""),
code("""fig, axs = plt.subplots(1, 3, figsize=(15,4.2))
ConfusionMatrixDisplay.from_predictions(te.y, (p>=thr).astype(int), display_labels=['normal','heatwave'], ax=axs[0], cmap='Reds', colorbar=False)
PrecisionRecallDisplay.from_predictions(te.y, p, ax=axs[1], name='SVM-RBF', color='#C8432B')
RocCurveDisplay.from_predictions(te.y, p, ax=axs[2], name='SVM-RBF', curve_kwargs={'color':'#C8432B'}); plt.tight_layout(); plt.show()"""),
code("""imp = pd.DataFrame(M['importance']).set_index('label')['mean'].sort_values()
imp.plot.barh(figsize=(7,4.5), color='#C8432B', title='Permutation importance (drop in PR-AUC)'); plt.show()"""),
md("## 6. Season replay — Delhi 2024"),
code("""s = te.assign(p=p)[(te.city=='Delhi') & (te.target_year==2024)]
fig, (a, b) = plt.subplots(2, 1, figsize=(12,5.5), sharex=True, gridspec_kw={'height_ratios':[2,1]})
a.plot(s.target_date, s.tmax_next, c='k', label='Observed Tmax'); a.plot(s.target_date, s.tmax_next - s.dep_next, ':', c='grey', label='Normal')
a.axhline(40, ls='--', c='#DE6E12'); a.fill_between(s.target_date, 20, 48, where=s.y==1, color='#C8432B', alpha=.15, label='Heatwave day')
a.set_ylim(20,48); a.legend(loc='upper left'); a.set_ylabel('°C')
b.bar(s.target_date, s.p, color=np.where(s.p>=thr,'#C8432B','#9AA5B1')); b.axhline(thr, ls='--', c='k'); b.set_ylabel('P(heatwave)')
plt.tight_layout(); plt.show()"""),
md("""## 7. Conclusions
* The RBF-kernel SVM gives the best F1 and recall on the unseen 2021–2024 seasons and catches most severe heatwave days.
* Linear models rank days almost as well, because the engineered *IMD margin* features already linearise much of the problem. The kernel's gain is mainly a better operating point.
* The first day of a heatwave spell is the hardest case. Adding NWP forecast inputs is the next step.

See the website (`uvicorn app.main:app`) for the interactive kernel lab and live predictions."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
here = Path(__file__).parent
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(here)}})
nbf.write(nb, here / "heatwave_svm_rbf.ipynb")
print("written")
