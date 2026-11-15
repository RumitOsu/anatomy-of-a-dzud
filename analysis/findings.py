"""Prints every headline number used on the page, from aimag_panel3.parquet. Run after build3.py."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import model as m, model2 as m2

A = pd.read_parquet("aimag_panel3.parquet")
a = A[(A.year >= 1994) & (A.code != "511") & A.rate.notna()].copy()   # 511 = the capital
a["dis"] = (a.rate >= 0.10).astype(int)
a["dry"] = a.sum_pr_z < -0.5; a["snowy"] = a.d_snowdays_z > 0.5; a["cold"] = a.s_win_t_z < -0.5
a["early"] = a.d_nd_snowdays_z > 0.5
a["k"] = a.dry.astype(int) + a.snowy + a.cold
print("province-winters:", len(a), "| mass die-offs:", int(a.dis.sum()), "| years:", a.year.min(), "-", a.year.max())

print("\n-- by number of conditions met")
print(a.groupby("k").dis.agg(n="size", die_offs="sum", share="mean").round(3).to_string())
print("\n-- by exact combination")
print(a.groupby(["dry", "snowy", "cold"]).dis.agg(n="size", die_offs="sum", share="mean").round(3).to_string())
print("\n-- what is known when")
print("after summer:\n", a.groupby("dry").dis.agg(n="size", die_offs="sum", share="mean").round(3).to_string())
print("by New Year:\n", a.groupby(["dry", "early"]).dis.agg(n="size", die_offs="sum", share="mean").round(3).to_string())

print("\n-- does the pattern survive a different cut-off? (share with a die-off, none vs all three)")
for th in (0.25, 0.5, 0.75, 1.0):
    k = (a.sum_pr_z < -th).astype(int) + (a.d_snowdays_z > th) + (a.s_win_t_z < -th)
    g = a.groupby(k).dis.mean()
    print(f"  {th:.2f} sd: {g.get(0, np.nan):.3f} -> {g.get(3, np.nan):.3f}")

print("\n-- uncertainty: resample whole years, 2000 times")
rng = np.random.default_rng(1); ys = a.year.unique(); bs = []
for _ in range(2000):
    s = pd.concat([a[a.year == y] for y in rng.choice(ys, len(ys))])
    bs.append(s.groupby("k").dis.mean().reindex(range(4)))
bs = pd.DataFrame(bs); ratio = bs[3] / bs[0].replace(0, np.nan)
print("  risk ratio, all three vs none: 5% / median / 95% =", ratio.quantile([.05, .5, .95]).round(1).tolist())

print("\n-- forecasting, every score on years the model never saw (leave one year out)")
X = a[["sum_pr_z", "d_snowdays_z", "s_win_t_z"]].clip(-4, 4); p = np.zeros(len(a))
for Y in a.year.unique():
    tr = (a.year != Y).values
    p[~tr] = LogisticRegression(C=1).fit(X[tr], a.dis[tr]).predict_proba(X[~tr])[:, 1]
print("  three conditions, logistic regression: AUC", round(roc_auc_score(a.dis, p), 3))
B = m.prep(A); B = B[(B.code != "511") & (B.year >= 1994) & B.rate.notna() & B.rate_prev2.notna()]
feats = m2.HIND + [c for c in B.columns if c.startswith("d_") and c.endswith("_z")]
o = m.evaluate(B, feats, "gbm", "loyo")
print(f"  gradient-boosted model, {len(feats)} inputs: AUC", round(roc_auc_score((o.rate >= .1).astype(int), o.p), 3))

print("\n-- crowding (herd above its own 10-year average by 15%+), within each condition count")
a["crowded"] = a.dens_rel > 1.15
print(a.groupby(["k", "crowded"]).dis.agg(n="size", share="mean").round(3).to_string())
print("\n-- provinces meeting all three, by year")
print(a[a.k == 3].groupby("year").dis.agg(n="size", die_offs="sum").T.to_string())
