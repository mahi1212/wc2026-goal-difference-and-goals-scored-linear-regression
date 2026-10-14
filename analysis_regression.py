"""
Objective 2 - linear regression analysis (FIFA World Cup 2026)
  2.1  goal_diff     ~ 8 pre-match variables   (lr21_match_dataset.csv, n=104)
  2.2  goals_scored  ~ 8 pre-match variables   (lr22_teammatch_dataset.csv, n=208)

Run:  python analysis_regression.py      (needs pandas, numpy, scipy, statsmodels, matplotlib)
Outputs: results.json, fig_models.png, fig_corr.png and a console summary.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import RepeatedKFold

SEED = 42
rng = np.random.default_rng(SEED)

lr21 = pd.read_csv("lr21_match_dataset.csv")
lr22 = pd.read_csv("lr22_teammatch_dataset.csv")

SPECS = {
    "2.1": dict(df=lr21, y="goal_diff", ids=["match_id", "home_team", "away_team"]),
    "2.2": dict(df=lr22, y="goals_scored", ids=["match_id", "team", "opponent"]),
}


def backward(X, y, alpha=0.10):
    """Backward elimination on p-values (used inside CV folds to avoid leakage)."""
    cols = list(X.columns)
    while cols:
        r = sm.OLS(y, sm.add_constant(X[cols])).fit()
        p = r.pvalues.drop("const")
        if p.max() <= alpha or len(cols) == 1:
            return cols
        cols.remove(p.idxmax())
    return cols


def metrics(y, p):
    resid = y - p
    ss_res, ss_tot = np.sum(resid ** 2), np.sum((y - y.mean()) ** 2)
    return dict(rmse=float(np.sqrt(np.mean(resid ** 2))), mae=float(np.mean(np.abs(resid))),
                r2=float(1 - ss_res / ss_tot))


def cv_eval(X, y, reps=20):
    """Repeated 5-fold CV for: baseline (train mean), full OLS, reduced OLS (selection inside folds)."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=reps, random_state=SEED)
    out = {"baseline": [], "full": [], "reduced": []}
    for tr, te in rkf.split(X):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        out["baseline"].append(metrics(yte.values, np.full(len(yte), ytr.mean())))
        full = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
        out["full"].append(metrics(yte.values, full.predict(sm.add_constant(Xte, has_constant="add")).values))
        cols = backward(Xtr, ytr)
        red = sm.OLS(ytr, sm.add_constant(Xtr[cols])).fit()
        out["reduced"].append(metrics(yte.values, red.predict(sm.add_constant(Xte[cols], has_constant="add")).values))
    return {k: {m: float(np.mean([d[m] for d in v])) for m in ("rmse", "mae", "r2")} for k, v in out.items()}


def oof_predictions(X, y):
    """One out-of-fold prediction per row (5-fold) for the actual-vs-predicted plot."""
    pred = np.zeros(len(y))
    idx = rng.permutation(len(y))
    for fold in np.array_split(idx, 5):
        tr = np.setdiff1d(np.arange(len(y)), fold)
        m = sm.OLS(y.iloc[tr], sm.add_constant(X.iloc[tr])).fit()
        pred[fold] = m.predict(sm.add_constant(X.iloc[fold], has_constant="add")).values
    return pred


results, fitted_store = {}, {}
for name, s in SPECS.items():
    df, ycol = s["df"], s["y"]
    X = df.drop(columns=s["ids"] + [ycol])
    y = df[ycol]
    Xc = sm.add_constant(X)
    ols = sm.OLS(y, Xc).fit()
    cluster = df["match_id"] if name == "2.2" else None
    robust = (ols.get_robustcov_results(cov_type="cluster", groups=cluster.values)
              if cluster is not None else ols.get_robustcov_results(cov_type="HC3"))
    red_cols = backward(X, y)
    red = sm.OLS(y, sm.add_constant(X[red_cols])).fit()
    resid = ols.resid
    bp = het_breuschpagan(resid, Xc)
    infl = ols.get_influence()
    cook = infl.cooks_distance[0]
    res = dict(
        n=int(len(df)),
        full=dict(r2=ols.rsquared, adj_r2=ols.rsquared_adj, f=float(ols.fvalue), f_p=float(ols.f_pvalue),
                  rmse_in=float(np.sqrt(np.mean(resid ** 2))),
                  coef={c: dict(b=float(ols.params[c]), se=float(ols.bse[c]), p=float(ols.pvalues[c]),
                                p_robust=float(robust.pvalues[i]),
                                lo=float(ols.conf_int().loc[c, 0]), hi=float(ols.conf_int().loc[c, 1]))
                        for i, c in enumerate(Xc.columns)}),
        reduced=dict(vars=red_cols, r2=red.rsquared, adj_r2=red.rsquared_adj,
                     coef={c: dict(b=float(red.params[c]), p=float(red.pvalues[c])) for c in red.params.index}),
        vif={c: float(vif(Xc.values, i)) for i, c in enumerate(Xc.columns) if c != "const"},
        shapiro_p=float(stats.shapiro(resid)[1]),
        bp_p=float(bp[1]),
        dw=float(durbin_watson(resid)),
        max_cook=float(cook.max()), n_cook_gt_4n=int((cook > 4 / len(df)).sum()),
        cv=cv_eval(X, y),
        std_beta={c: float(ols.params[c] * X[c].std() / y.std()) for c in X.columns},
    )
    results[name] = res
    fitted_store[name] = dict(fitted=ols.fittedvalues.values, resid=resid.values, y=y.values,
                              oof=oof_predictions(X, y))

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
for r, (name, lab) in enumerate((("2.1", "goal difference"), ("2.2", "goals scored"))):
    d = fitted_store[name]
    a = ax[r, 0]
    a.scatter(d["oof"], d["y"], s=14, alpha=0.65, color="#1f77b4", edgecolor="none")
    lim = [min(d["oof"].min(), d["y"].min()) - 0.3, max(d["oof"].max(), d["y"].max()) + 0.3]
    a.plot(lim, lim, "--", color="grey", lw=1)
    a.set_xlabel(f"Predicted {lab} (5-fold CV)", fontsize=8)
    a.set_ylabel(f"Actual {lab}", fontsize=8)
    a.set_title(f"Model {name}: predicted vs actual", fontsize=9)
    b = ax[r, 1]
    b.scatter(d["fitted"], d["resid"], s=14, alpha=0.65, color="#d62728", edgecolor="none")
    b.axhline(0, color="grey", lw=1)
    b.set_xlabel("Fitted value", fontsize=8)
    b.set_ylabel("Residual", fontsize=8)
    b.set_title(f"Model {name}: residuals vs fitted", fontsize=9)
    for q in (a, b):
        q.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig("fig_models.png", dpi=200)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
for q, (name, s) in zip(ax, SPECS.items()):
    df = s["df"]
    cols = [c for c in df.columns if c not in s["ids"]]
    cm = df[cols].corr()
    im = q.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    q.set_xticks(range(len(cols)))
    q.set_yticks(range(len(cols)))
    short = [c.replace("_avg", "").replace("_diff", "Δ")[:13] for c in cols]
    q.set_xticklabels(short, rotation=75, fontsize=5.5, ha="right")
    q.set_yticklabels(short, fontsize=5.5)
    q.set_title(f"Model {name}: correlation matrix", fontsize=8)
plt.tight_layout()
plt.savefig("fig_corr.png", dpi=200)
plt.close()

# ---------------------------------------------------------------- console summary
for name, r in results.items():
    print(f"\n=== Model {name} (n={r['n']}) ===")
    print(f"R2={r['full']['r2']:.3f} adjR2={r['full']['adj_r2']:.3f} F p={r['full']['f_p']:.2g}")
    for c, v in r["full"]["coef"].items():
        print(f"  {c:26s} b={v['b']:8.3f} p={v['p']:.3f} p_robust={v['p_robust']:.3f}")
    print("  reduced:", r["reduced"]["vars"], f"adjR2={r['reduced']['adj_r2']:.3f}")
    print("  CV:", {k: {m: round(x, 3) for m, x in v.items()} for k, v in r["cv"].items()})
    print(f"  Shapiro p={r['shapiro_p']:.3f} BP p={r['bp_p']:.3f} DW={r['dw']:.2f} maxVIF={max(r['vif'].values()):.1f}")
