"""10: Restricted to participants with linked primary care records (p42040 GP clinical event records > 0, ~45% of participants),
in whom diabetes ascertainment includes primary care; matching is repeated within this subset."""
import pandas as pd, numpy as np
from common import *
M = os.environ.get("UKB_DATA_DIR", "../ukb_data")
g = pd.read_csv(f"{M}/UKB619657_health_outcomes_2025Nov.csv", usecols=["eid", "p42040"], low_memory=False)
df = load().merge(g, on="eid", how="left"); df = df[pd.to_numeric(df.p42040, errors="coerce") > 0]
base = base_cohort(df); cases = exposed(base, "AP"); m = follow(match(base, cases, seed=21))
rows = []
for lab, s in [("All", m)] + [(f"HbA1c {h}", m[m.hcat == h]) for h in HLAB]:
    r = cox(s); rows.append(dict(group=lab, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), CIF5=f"{cif(s[s.exp==1]):.1f} vs {cif(s[s.exp==0]):.1f}"))
R = pd.DataFrame(rows); print(f"Primary-care-linked subset: baseline {len(base):,}, AP {len(cases):,}"); print(R.to_string(index=False))
R.to_csv(os.path.join(RES, "F11_gp_linked.csv"), index=False, encoding="utf-8-sig")
