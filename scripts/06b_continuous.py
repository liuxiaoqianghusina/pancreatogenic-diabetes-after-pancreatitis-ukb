"""06c: Effect modification by continuous HbA1c (replaces the unstable unstratified spline in 06). Matched-set-stratified Cox:
exp + h + exp×h with h = (HbA1c − 36)/5; a quadratic interaction tests nonlinearity. Writes the AP HR curve over HbA1c 28–47 mmol/mol."""
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2
from common import *
m = pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl"))
X = m[["T", "E", "exp", "hba1c", "mid"]].copy(); X["E"] = (X.E == 1).astype(int)
X["h"] = (X.hba1c - 36) / 5; X["eh"] = X.exp * X.h
f = CoxPHFitter().fit(X[["T", "E", "exp", "h", "eh", "mid"]], "T", "E", strata=["mid"])
b, V = f.params_, f.variance_matrix_; idx = ["exp", "eh"]
grid = np.arange(28, 47.5, 0.5); rows = []
for hv in grid:
    g = np.array([1, (hv - 36) / 5]); mu = g @ b[idx].values; se = np.sqrt(g @ V.loc[idx, idx].values @ g)
    rows.append(dict(hba1c=hv, HR=np.exp(mu), lo=np.exp(mu - 1.96 * se), hi=np.exp(mu + 1.96 * se)))
G = pd.DataFrame(rows); G.to_csv(os.path.join(RES, "F06_spline_HR.csv"), index=False)
print(f"Per 5 mmol/mol HbA1c, the AP HR is multiplied by {np.exp(b['eh']):.2f} ({np.exp(b['eh']-1.96*np.sqrt(V.loc['eh','eh'])):.2f}–{np.exp(b['eh']+1.96*np.sqrt(V.loc['eh','eh'])):.2f}), P = {f.summary.loc['eh','p']:.4f}")
print("HR at HbA1c 30/33/36/40/45:", {h: f"{G.loc[G.hba1c==h,'HR'].item():.2f} ({G.loc[G.hba1c==h,'lo'].item():.2f}–{G.loc[G.hba1c==h,'hi'].item():.2f})" for h in [30, 33, 36, 40, 45]})
# nonlinearity: add a quadratic interaction
X["eh2"] = X.exp * X.h ** 2; X["h2"] = X.h ** 2
f1 = CoxPHFitter().fit(X[["T", "E", "exp", "h", "h2", "eh", "mid"]], "T", "E", strata=["mid"])
f2 = CoxPHFitter().fit(X[["T", "E", "exp", "h", "h2", "eh", "eh2", "mid"]], "T", "E", strata=["mid"])
lr = 2 * (f2.log_likelihood_ - f1.log_likelihood_)
print(f"Nonlinear interaction (exp × HbA1c², 1 df) likelihood-ratio P = {chi2.sf(lr, 1):.3f}")
