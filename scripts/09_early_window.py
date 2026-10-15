"""09: No blanking period (index date = AP date); HRs for days 0–90 and >90 separately, to quantify early detection around the admission."""
import numpy as np, pandas as pd
from common import *
df = load(); base = base_cohort(df); cases = exposed(base, "AP", blank=0)
m = follow(match(base, cases, seed=5)); rows = []
for a, b_, lab in [(0, 90 / 365.25, "days 0–90 after AP"), (90 / 365.25, 99, "after day 90")]:
    d = m[m["T"] > a].copy(); d["E"] = np.where(d["T"] <= b_, d.E, 0); d["T"] = d["T"].clip(upper=b_) - a; d = d[d["T"] > 0]
    r = cox(d); h = cox(d[d.hcat == "<36"])
    rows.append(dict(period=lab, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", incidence=f"{rate(d[d.exp==1]):.1f} vs {rate(d[d.exp==0]):.1f}", HR=fmt(r), HR_lt36=fmt(h)))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(os.path.join(RES, "F09_early_window.csv"), index=False, encoding="utf-8-sig")
