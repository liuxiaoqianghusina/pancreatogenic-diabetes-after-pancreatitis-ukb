"""07: Severity proxy of AP and diabetes risk (dose-response).
Severe = organ failure first recorded from 3 days before to 30 days after AP (N17 acute kidney injury, J96 respiratory failure, R57 shock,
R65 SIRS) or pancreatic pseudocyst (K86.3) first recorded by the index date (90 days). p41280 holds only the first date of each code,
so patients with the code before AP are classified as non-severe (conservative)."""
import numpy as np, pandas as pd
from common import *

m = pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl"))
x = pd.read_pickle(os.path.join(ROOT, "cache", "extra_codes.pkl"))
cs = m[m.exp == 1][["eid", "t_evt", "t0"]].merge(x, on="eid", how="left")
win = lambda c, a, b: cs[c].notna() & (cs[c] >= cs.t_evt + pd.Timedelta(days=a)) & (cs[c] <= cs.t_evt + pd.Timedelta(days=b))
cs["of"] = win("h_aki", -3, 30) | win("h_resp", -3, 30) | win("h_shock", -3, 30) | win("h_sirs", -3, 30)
cs["pseudo"] = win("h_pseudo", -3, 90)
cs["sev"] = np.select([cs["of"] & cs.pseudo, cs["of"] | cs.pseudo], ["organ failure + pseudocyst", "organ failure or pseudocyst"], "none")
cs["severe"] = (cs["of"] | cs.pseudo).astype(int)
print("AP patients: organ failure", int(cs["of"].sum()), "; pseudocyst", int(cs.pseudo.sum()), "; either", int(cs.severe.sum()), f"({cs.severe.mean()*100:.1f}%)")
m["severe"] = m.mid.map(m[m.exp == 1].set_index("mid").eid.map(cs.set_index("eid").severe))
rows = []
for lab, s in [("Non-severe AP vs matched controls", m[m.severe == 0]), ("Severe-proxy AP vs matched controls", m[m.severe == 1])]:
    r = cox(s); a, c = s[s.exp == 1], s[s.exp == 0]
    rows.append(dict(group=lab, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), CIF5=f"{cif(a):.1f} vs {cif(c):.1f}"))
# within AP: severe vs non-severe (adjusted for age, sex, HbA1c, BMI)
a = covars(m[m.exp == 1].copy()); a["age_evt"] = (a.t_evt - a.birth).dt.days / 365.25
r = cox(a.rename(columns={"severe": "x"}), x="x", covs=["age_evt", "male", "hba1c", "bmi"], strata=None)
rows.append(dict(group="Within AP: severe proxy vs non-severe", AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), CIF5=""))
# interaction: does severity modify the AP effect
mm = m.dropna(subset=["severe"]).copy(); mm["xs"] = mm.exp * mm.severe
from lifelines import CoxPHFitter
f = CoxPHFitter().fit(mm[["T", "E", "exp", "xs", "mid"]].assign(E=(mm.E == 1).astype(int)), "T", "E", strata=["mid"])
R = pd.DataFrame(rows); print(R.to_string(index=False)); print(f"Severity × AP interaction P = {f.summary.loc['xs','p']:.3f}")
R.to_csv(os.path.join(RES, "F07_severity.csv"), index=False, encoding="utf-8-sig")
