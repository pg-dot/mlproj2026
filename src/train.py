"""
Train the Heatwave Early-Warning classifier: Support Vector Machine with RBF kernel.

    python src/train.py

Outputs
  models/svm_rbf.joblib     sklearn Pipeline(StandardScaler -> SVC(kernel='rbf'))
  models/metrics.json       all evaluation numbers
  web/data/model.json       exported model for exact in-browser inference
  web/data/results.json     curves, tables, replay series for the website
  web/data/kernel_demo.json 2-D decision boundaries for the interactive kernel explainer
"""
import json, time, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, f1_score, precision_score,
                             recall_score, confusion_matrix, brier_score_loss, fbeta_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import (load_raw, climatology, build_features, split, FEATURES, FEATURE_LABELS,
                      CITIES, COASTAL, TRAIN_END)

RNG = 42
OUT_W = ROOT / "web" / "data"; OUT_W.mkdir(parents=True, exist_ok=True)
OUT_M = ROOT / "models"; OUT_M.mkdir(exist_ok=True)


def r(x, n=4):
    return float(round(float(x), n))


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(roc_auc=r(roc_auc_score(y, p)), pr_auc=r(average_precision_score(y, p)),
                f1=r(f1_score(y, yhat, zero_division=0)), f2=r(fbeta_score(y, yhat, beta=2, zero_division=0)),
                precision=r(precision_score(y, yhat, zero_division=0)),
                recall=r(recall_score(y, yhat, zero_division=0)),
                specificity=r(tn / (tn + fp)), brier=r(brier_score_loss(y, p)),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), threshold=r(thr))


def best_threshold(y, p):
    """Threshold maximising F1 on validation data (searched over a fine grid)."""
    grid = np.linspace(0.05, 0.95, 181)
    f = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def curve_pts(x, y, n=120):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [[r(x[i], 4), r(y[i], 4)] for i in idx]


def main():
    t0 = time.time()
    raw = load_raw(ROOT / "data" / "india_2000_2024_daily_weather.csv")
    normals = climatology(raw)
    df = build_features(raw, normals)
    tr, te = split(df)
    Xtr, ytr, gtr = tr[FEATURES].values, tr.y.values, tr.target_year.values
    Xte, yte = te[FEATURES].values, te.y.values
    print(f"rows dev(2000-20)={len(tr)} (+{ytr.sum()})  test(2021-24)={len(te)} (+{yte.sum()})")

    # ---------- 1. Hyper-parameter search: C, gamma, positive-class weight ----------
    # Year-blocked 5-fold CV on 2000-2020: whole years are held out together so that
    # consecutive (autocorrelated) days never sit on both sides of a split.
    # Score = PR-AUC (average precision), the right metric for a rare-event (~1.3%) problem.
    cv = GroupKFold(n_splits=5)
    Cs = [0.3, 1, 3, 10, 30]
    gammas = [0.003, 0.01, 0.03, 0.1, 0.3]
    weights = [{0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 10}]
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", cache_size=2000))])
    gs = GridSearchCV(pipe, {"svm__C": Cs, "svm__gamma": gammas, "svm__class_weight": weights},
                      scoring="average_precision", cv=cv, n_jobs=-1, refit=False)
    gs.fit(Xtr, ytr, groups=gtr)
    res = pd.DataFrame(gs.cv_results_)
    bp = gs.best_params_
    bestC, bestG, bestW = bp["svm__C"], bp["svm__gamma"], bp["svm__class_weight"]
    sel = res[res.param_svm__class_weight.apply(lambda w: w == bestW)]
    grid = [[r(sel[(sel.param_svm__C == c) & (sel.param_svm__gamma == g)].mean_test_score.iloc[0])
             for g in gammas] for c in Cs]
    wscores = {str(w[1]): r(res[res.param_svm__class_weight.apply(lambda x: x == w)].mean_test_score.max())
               for w in weights}
    print(f"grid search {time.time()-t0:.0f}s  best C={bestC} gamma={bestG} w={bestW} CV PR-AUC={gs.best_score_:.3f}")

    # ---------- 2. Decision threshold from out-of-fold probabilities ----------
    def make_svm():
        return Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel="rbf", C=bestC, gamma=bestG, class_weight=bestW,
                                     probability=True, random_state=RNG, cache_size=2000))])
    oof = cross_val_predict(make_svm(), Xtr, ytr, groups=gtr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr = best_threshold(ytr, oof)

    # ---------- 3. Final SVM-RBF on all development years, Platt-scaled probabilities ----------
    model = make_svm().fit(Xtr, ytr)
    pte = model.predict_proba(Xte)[:, 1]
    print(f"svm fitted {time.time()-t0:.0f}s  n_SV={model[-1].n_support_.tolist()} thr={thr:.2f}")

    # ---------- 4. Baselines (same features, same CV-chosen threshold procedure) ----------
    def fit_prob(est):
        p = Pipeline([("scaler", StandardScaler()), ("m", est)])
        o = cross_val_predict(p, Xtr, ytr, groups=gtr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        return o, p.fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    baselines = {}
    lr_oof, lr_te = fit_prob(LogisticRegression(class_weight=bestW, max_iter=3000))
    baselines["Logistic Regression"] = (lr_oof, lr_te)
    baselines["Linear SVM (no kernel)"] = fit_prob(CalibratedClassifierCV(
        LinearSVC(class_weight=bestW, C=1.0, max_iter=50000), cv=3, method="sigmoid"))
    # persistence: "tomorrow will be like today"
    baselines["Persistence (today = tomorrow)"] = (tr.hw_today.values.astype(float), te.hw_today.values.astype(float))
    # IMD rule applied to today's anomaly carried forward to tomorrow

    # onset = first day of a heatwave spell (today is NOT a heatwave, tomorrow is).
    # This is the hardest and most valuable case for an early-warning system.
    onset = (yte == 1) & (te.hw_today.values == 0)
    def onset_recall(p, t):
        return r(((p >= t) & onset).sum() / onset.sum())

    comp = [dict(model="SVM — RBF kernel (ours)", **metrics(yte, pte, thr), onset_recall=onset_recall(pte, thr))]
    for name, (bo, bte) in baselines.items():
        bt = 0.5 if set(np.unique(bo)) <= {0.0, 1.0} else best_threshold(ytr, bo)
        comp.append(dict(model=name, **metrics(yte, bte, bt), onset_recall=onset_recall(bte, bt)))
    print("onset days in test:", int(onset.sum()))
    for c in comp:
        print(f"  {c['model']:<40} ROC={c['roc_auc']:.3f} PR={c['pr_auc']:.3f} F1={c['f1']:.3f} "
              f"P={c['precision']:.3f} R={c['recall']:.3f} onset={c['onset_recall']:.3f}")

    # ---------- 4. Diagnostics ----------
    fpr, tpr, _ = roc_curve(yte, pte); prec, rec, _ = precision_recall_curve(yte, pte)
    lfpr, ltpr, _ = roc_curve(yte, lr_te); lprec, lrec, _ = precision_recall_curve(yte, lr_te)
    yhat = (pte >= thr).astype(int)
    te = te.assign(p=pte, yhat=yhat)

    per_city = []
    for c in CITIES:
        s = te[te.city == c]
        per_city.append(dict(city=c, coastal=int(c in COASTAL), events=int(s.y.sum()),
                             detected=int(((s.y == 1) & (s.yhat == 1)).sum()),
                             false_alarms=int(((s.y == 0) & (s.yhat == 1)).sum()),
                             pr_auc=r(average_precision_score(s.y, s.p)) if s.y.sum() else None))
    per_year = []
    for yv, s in te.groupby("target_year"):
        per_year.append(dict(year=int(yv), events=int(s.y.sum()),
                             detected=int(((s.y == 1) & (s.yhat == 1)).sum()),
                             false_alarms=int(((s.y == 0) & (s.yhat == 1)).sum()),
                             recall=r(recall_score(s.y, s.yhat, zero_division=0))))
    # severe events caught
    sev = te[te.y_severe == 1]
    severe_recall = r(sev.yhat.mean()) if len(sev) else None

    # calibration (reliability) bins
    bins = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pte >= lo) & (pte < hi)
        if m.sum() >= 5:
            calib.append(dict(lo=r(lo, 2), hi=r(min(hi, 1), 2), n=int(m.sum()),
                              mean_p=r(pte[m].mean()), observed=r(yte[m].mean())))

    # permutation importance (drop in PR-AUC on test)
    sub = te.sample(n=min(6000, len(te)), random_state=RNG)
    sub = pd.concat([sub, te[te.y == 1]]).drop_duplicates()
    pim = permutation_importance(model, sub[FEATURES].values, sub.y.values, scoring="average_precision",
                                 n_repeats=5, random_state=RNG, n_jobs=-1)
    importance = sorted([dict(feature=f, label=FEATURE_LABELS[f], mean=r(m), std=r(s))
                         for f, m, s in zip(FEATURES, pim.importances_mean, pim.importances_std)],
                        key=lambda d: -d["mean"])
    print(f"diagnostics {time.time()-t0:.0f}s; top features:", [d['feature'] for d in importance[:5]])

    # replay series for website: whole test period, in season, per city
    replay = {}
    for c in CITIES:
        s = te[te.city == c]
        replay[c] = dict(date=s.target_date.dt.strftime("%Y-%m-%d").tolist(),
                         tmax=[r(v, 1) for v in s.tmax_next], normal=[r(v, 1) for v in s.tmax_next - s.dep_next],
                         y=s.y.astype(int).tolist(), sev=s.y_severe.astype(int).tolist(),
                         p=[r(v, 3) for v in s.p])

    # example feature rows for the predictor presets (real test-set days)
    presets = []
    picks = [("Delhi", "2024-05-29", "Delhi — 29 May 2024"),
             ("Lucknow", "2024-06-18", "Lucknow — 18 Jun 2024"),
             ("Kolkata", "2024-04-30", "Kolkata — 30 Apr 2024 (coastal)"),
             ("Jaipur", "2022-05-15", "Jaipur — 15 May 2022"),
             ("Ahmedabad", "2024-05-24", "Ahmedabad — 24 May 2024"),
             ("Bangalore", "2024-04-15", "Bangalore — 15 Apr 2024"),
             ("Mumbai", "2023-06-10", "Mumbai — 10 Jun 2023")]
    for c, d, lab in picks:
        h = df[(df.city == c) & (df.target_date <= pd.Timestamp(d))].tail(3)  # rows for t-2, t-1, t
        row = h.iloc[-1]
        if row.target_date != pd.Timestamp(d):
            continue
        presets.append(dict(label=lab, city=c, date=d, today=row.date.strftime("%Y-%m-%d"),
                            actual=int(row.y), severe=int(row.y_severe), tmax_next=r(row.tmax_next, 1),
                            inp=dict(tmax=r(row.tmax, 1), tmax_1=r(h.tmax.iloc[1], 1), tmax_2=r(h.tmax.iloc[0], 1),
                                     tmin=r(row.tmin, 1), app_tmax=r(row.app_tmax, 1),
                                     rain7=r(np.expm1(row.rain7_log), 1), wind_max=r(row.wind_max, 1))))

    # ---------- 5. Exports ----------
    joblib.dump(model, OUT_M / "svm_rbf.joblib")
    sc, svm = model[0], model[-1]
    model_json = dict(
        kernel="rbf", C=bestC, gamma=float(svm._gamma), features=FEATURES, feature_labels=FEATURE_LABELS,
        scaler_mean=[r(v, 6) for v in sc.mean_], scaler_scale=[r(v, 6) for v in sc.scale_],
        support_vectors=[[r(v, 5) for v in sv] for sv in svm.support_vectors_],
        dual_coef=[r(v, 6) for v in svm.dual_coef_[0]], intercept=r(svm.intercept_[0], 6),
        probA=r(svm.probA_[0], 6), probB=r(svm.probB_[0], 6), threshold=r(thr, 3),
        n_support=svm.n_support_.tolist(),
        normals={c: [r(v, 2) for v in normals[c]] for c in CITIES},
        cities={c: dict(lat=v[0], lon=v[1], coastal=v[2]) for c, v in CITIES.items()},
    )
    (OUT_W / "model.json").write_text(json.dumps(model_json, separators=(",", ":")))

    # verification vector: the website & API must reproduce these probabilities exactly
    ver = te.sample(25, random_state=1)
    (OUT_M / "verification.json").write_text(json.dumps(dict(
        X=ver[FEATURES].values.round(6).tolist(), p=model.predict_proba(ver[FEATURES].values)[:, 1].tolist(),
        decision=model.decision_function(ver[FEATURES].values).tolist())))

    counts = dict(rows_total=int(len(raw)), cities=len(CITIES), years="2000–2024",
                  train=dict(rows=len(tr), pos=int(ytr.sum()), years="2000–2020"),
                  test=dict(rows=len(te), pos=int(yte.sum()), years="2021–2024"),
                  heatwave_days_all=int(df.hw_today.sum()), severe_days_all=int(df.severe_today.sum()),
                  by_city=df.groupby("city").hw_today.sum().astype(int).to_dict(),
                  by_year=df.groupby(df.date.dt.year).hw_today.sum().astype(int).to_dict(),
                  by_month=df.groupby(df.date.dt.month).hw_today.sum().astype(int).to_dict())
    results = dict(
        best=dict(C=bestC, gamma=bestG, pos_weight=bestW[1], cv_pr_auc=r(gs.best_score_), threshold=r(thr, 3),
                  n_support=svm.n_support_.tolist(), weight_scores=wscores,
                  oof_pr_auc=r(average_precision_score(ytr, oof)), oof_roc_auc=r(roc_auc_score(ytr, oof))),
        grid=dict(C=Cs, gamma=gammas, pr_auc=grid),
        comparison=comp, severe_recall=severe_recall, onset_days=int(onset.sum()),
        roc=dict(svm=curve_pts(fpr, tpr), lr=curve_pts(lfpr, ltpr)),
        pr=dict(svm=curve_pts(rec[::-1], prec[::-1]), lr=curve_pts(lrec[::-1], lprec[::-1]),
                base_rate=r(yte.mean())),
        per_city=per_city, per_year=per_year, calibration=calib, importance=importance,
        counts=counts, replay=replay, presets=presets,
    )
    (OUT_W / "results.json").write_text(json.dumps(results, separators=(",", ":")))
    (OUT_M / "metrics.json").write_text(json.dumps({k: results[k] for k in
                                       ["best", "grid", "comparison", "severe_recall", "onset_days", "per_city", "per_year",
                                        "calibration", "importance", "counts"]}, indent=2))

    # ---------- 6. 2-D kernel explainer data ----------
    kernel_demo(tr)
    import runpy  # attach city/date to each support vector for the website's explanations
    runpy.run_path(str(ROOT / "src" / "export_sv_meta.py"))
    runpy.run_path(str(ROOT / "src" / "export_history.py"))  # daily data for the website date picker
    print(f"done in {time.time()-t0:.0f}s")


def kernel_demo(tr):
    """Train small 2-feature SVMs (today's departure vs. today's Tmax gap to threshold)
    over a grid of C and gamma so the website can show how the RBF boundary bends."""
    s = tr[tr.target_month.between(4, 6)]
    pos = s[s.y == 1].sample(n=min(140, int(s.y.sum())), random_state=RNG)
    neg = s[s.y == 0].sample(n=360, random_state=RNG)
    d = pd.concat([pos, neg])
    X = d[["dep", "thr_gap"]].values; y = d.y.values
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    xs = np.linspace(-9, 11, 64); ys = np.linspace(-14, 8, 64)
    gx, gy = np.meshgrid(xs, ys)
    G = (np.c_[gx.ravel(), gy.ravel()] - mu) / sd
    Cs = [0.1, 1, 10, 100]; gammas = [0.05, 0.3, 1, 3, 10]
    combos = {}
    for C in Cs:
        for g in gammas:
            m = SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced").fit(Z, y)
            f = m.decision_function(G).reshape(gx.shape)
            acc = float((m.predict(Z) == y).mean())
            combos[f"{C}|{g}"] = dict(f=[[round(float(v), 2) for v in row] for row in f],
                                      sv=m.support_.tolist(), acc=round(acc, 3), n_sv=int(len(m.support_)))
    lin = SVC(kernel="linear", C=1, class_weight="balanced").fit(Z, y)
    combos["linear"] = dict(f=[[round(float(v), 2) for v in row] for row in lin.decision_function(G).reshape(gx.shape)],
                            sv=lin.support_.tolist(), acc=round(float((lin.predict(Z) == y).mean()), 3),
                            n_sv=int(len(lin.support_)))
    out = dict(x=[round(float(v), 3) for v in xs], y=[round(float(v), 3) for v in ys],
               points=[[round(float(a), 2), round(float(b), 2), int(c)] for (a, b), c in zip(X, y)],
               mu=mu.tolist(), sd=sd.tolist(), C=Cs, gamma=gammas, combos=combos,
               xlabel="Today's departure from normal (°C)", ylabel="Today's Tmax minus IMD threshold (°C)")
    (OUT_W / "kernel_demo.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
