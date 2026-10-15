"""03: Sensitivity analyses and negative-control outcomes. Each analysis reports the overall HR, the HRs at HbA1c <36 and 42–47 mmol/mol,
and the P value for the multiplicative AP × HbA1c interaction."""
import numpy as np, pandas as pd
from common import *

df = load()
rows = []
def summarize(lab, m):
    r = cox(m); a = cox(m[m.hcat == "<36"]); b = cox(m[m.hcat == "42-47"])
    mm = m.copy(); mm["xh"] = mm.exp * (mm.hba1c - 36) / 5
    f = CoxPHFitter().fit(mm[["T", "E", "exp", "xh", "mid"]].assign(E=(mm.E == 1).astype(int)), "T", "E", strata=["mid"])
    rows.append(dict(analysis=lab, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR_all=fmt(r), HR_lt36=fmt(a), HR_42_47=fmt(b),
                     P_interaction=f"{f.summary.loc['xh','p']:.3f}"))
    print(rows[-1], flush=True)

def run(lab, ap_def="fo", dm_def="fo", blank=90, max_gap=None, post=None):
    base = base_cohort(df, ap_def, dm_def); cases = exposed(base, "AP", blank=blank, max_gap=max_gap)
    m = follow(match(base, cases))
    if post is not None: m = post(m)
    summarize(lab, m); return m

m0 = run("Main analysis")
run("S1 AP and diabetes from hospital records (HES) only", ap_def="hes", dm_def="hes")
run("S2 Diabetes from hospital records only", dm_def="hes")
run("S3 AP within 5 years of HbA1c measurement", max_gap=5)
run("S4 AP within 3 years of HbA1c measurement", max_gap=3)
run("S5 180-day blanking period", blank=180)
def drop_pcan2(m):
    bad = m.pcan.notna() & (m.pcan <= m.t0 + pd.Timedelta(days=730))
    m = m[~bad]; keep = m.groupby("mid").exp.transform(lambda s: (s == 1).any() and (s == 0).any()); return m[keep]
run("S6 Excluding pancreatic cancer within 2 years of the index date", post=drop_pcan2)
def drop_cp(m):
    # censor at a diagnosis of chronic pancreatitis during follow-up
    m = m.copy(); stop = m.t0 + pd.to_timedelta(m["T"] * 365.25, unit="D")
    c = m.h_cp.notna() & (m.h_cp > m.t0) & (m.h_cp < stop)
    m.loc[c, "E"] = 0; m.loc[c, "T"] = (m.loc[c, "h_cp"] - m.loc[c, "t0"]).dt.days / 365.25
    return m[m["T"] > 0]
run("S7 Censoring at chronic pancreatitis", post=drop_cp)
# S8 1:10 matching (multivariable adjustment is reported in 02)
base = base_cohort(df); cases = exposed(base, "AP")
m10 = follow(match(base, cases, k=10, seed=7)); summarize("S8 1:10 matching", m10)
# S9 first diabetes code E10 (type 1) not counted as the outcome (censored)
m9 = m0.copy(); m9.loc[(m9.E == 1) & (m9.dm_type == "e10"), "E"] = 0; summarize("S9 E10 codes not counted as outcome", m9)
pd.DataFrame(rows).to_csv(os.path.join(RES, "F03_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ---- negative-control outcomes: senile cataract (H25), knee osteoarthritis (M17) ----
nc = []
for col, lab in [("cat", "Senile cataract H25"), ("koa", "Knee osteoarthritis M17")]:
    d = m0.drop(columns=["E", "T"]).copy()
    d = d[~(d[col].notna() & (d[col] <= d.t0))]
    d = follow(d, outcome=col)
    keep = d.groupby("mid").exp.transform(lambda s: (s == 1).any() and (s == 0).any()); d = d[keep]
    r = cox(d); nc.append(dict(outcome=lab, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), P=f"{r['p']:.2f}"))
    print(nc[-1], flush=True)
pd.DataFrame(nc).to_csv(os.path.join(RES, "F03_negative_control.csv"), index=False, encoding="utf-8-sig")
m0.to_csv(os.path.join(ROOT, "cache", "matched_main_for_R.csv"), columns=["mid", "exp", "T", "E", "hba1c", "hcat", "age_evt", "male", "bmi"], index=False)
