"""05: Controls matched on sex and birth year only (the usual approach of previous database studies) vs. additionally on BMI and
pre-attack HbA1c, to show how much the conventional approach overestimates the association."""
import numpy as np, pandas as pd
import F_common as C
from common import *

df = load(); base = base_cohort(df); cases = exposed(base, "AP")
def match_on(keys, seed=20261011, k=5):
    rng = np.random.default_rng(seed)
    keep = ["eid", "p31", "p34", "hcat", "bcat", "d0", "DM", "dm", "death", "lost", "AP_any", "pcan", "h_cp"]
    grp = {key: g for key, g in base[keep].groupby(keys, observed=True)}
    recs = []
    for mid, (_, c) in enumerate(cases.iterrows()):
        g = grp.get(tuple(c[x] for x in keys)); t = c.t0
        if g is None: continue
        ok = g[(g.eid != c.eid) & (g.d0 < c.t_evt) & ~(g.DM.notna() & (g.DM <= t)) & ~(g.dm.notna() & (g.dm <= t)) & ~(g.death.notna() & (g.death <= t))
               & ~(g.lost.notna() & (g.lost <= t)) & ~(g.AP_any.notna() & (g.AP_any <= t)) & ~(g.pcan.notna() & (g.pcan <= t)) & ~(g.h_cp.notna() & (g.h_cp <= t))]
        if len(ok) == 0: continue
        sel = ok.eid.values[rng.choice(len(ok), min(k, len(ok)), replace=False)]
        recs.append((mid, 1, c.eid, t)); recs += [(mid, 0, s, t) for s in sel]
    m = pd.DataFrame(recs, columns=["mid", "exp", "eid", "t0"]).merge(base.drop(columns=[c for c in ["t0"] if c in base]), on="eid")
    return follow(m)
rows = []
for lab, keys in [("Previous approach: sex + birth year", ["p31", "p34"]), ("+ BMI category", ["p31", "p34", "bcat"]), ("This study: + HbA1c and BMI categories", ["p31", "p34", "hcat", "bcat"])]:
    m = match_on(keys); r = cox(m); a, c = m[m.exp == 1], m[m.exp == 0]
    rows.append(dict(matching=lab, AP_HbA1c_median=round(a.hba1c.median(), 1), ctrl_HbA1c_median=round(c.hba1c.median(), 1),
                     AP_BMI=round(a.bmi.mean(), 1), ctrl_BMI=round(c.bmi.mean(), 1), AP_prediabetes_pct=f"{(a.hba1c>=42).mean()*100:.1f}%",
                     ctrl_prediabetes_pct=f"{(c.hba1c>=42).mean()*100:.1f}%", incidence=f"{rate(a):.1f} vs {rate(c):.1f}", HR=fmt(r),
                     CIF5=f"{cif(a):.1f}% vs {cif(c):.1f}%"))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "F05_conventional_matching.csv"), index=False, encoding="utf-8-sig")
