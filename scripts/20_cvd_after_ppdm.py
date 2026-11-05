"""20: cardiovascular events after post-pancreatitis diabetes (feasibility count reported in the Discussion).
Incident diabetes after baseline; AP before diabetes; no MI or stroke before diabetes; no pancreatic cancer up to 1 year after diabetes.
MACE = first myocardial infarction (I21-I23), stroke (I60, I61, I63, I64) or cardiovascular death (primary cause I00-I99).
Writes results/H04_cvd_after_ppdm.txt and cache/cvd_dates.pkl."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
H = os.path.join(os.environ.get("UKB_DATA_DIR", "../ukb_data"), "UKB619657_health_outcomes_2025Nov.csv")
FO = {"mi21": "p131298", "mi22": "p131300", "mi23": "p131302", "sah": "p131360", "ich": "p131362", "isch": "p131366", "strk": "p131368", "hf": "p131354"}
cv = pd.read_csv(H, usecols=["eid", "p40001_i0"] + list(FO.values()), dtype=str)
cv["eid"] = cv.eid.astype(int)
for k, c in FO.items():
    d = pd.to_datetime(cv[c], errors="coerce"); cv[k] = d.mask(d.dt.year.isin([1900, 1901, 1902, 1903, 2037]))
cv["cvdeath"] = cv.p40001_i0.fillna("").str.match(r"^I")
cv = cv[["eid", "cvdeath"] + list(FO)]
cv.to_pickle(os.path.join(ROOT, "cache", "cvd_dates.pkl"))
df = load(); base = base_cohort(df).merge(cv, on="eid", how="left")
base["mi"] = base[["mi21", "mi22", "mi23"]].min(axis=1); base["stroke"] = base[["sah", "ich", "isch", "strk"]].min(axis=1)
base["cvdeath_date"] = base.death.where(base.cvdeath == True)
base["mace"] = base[["mi", "stroke", "cvdeath_date"]].min(axis=1)
inc = base[base.DM.notna() & (base.DM > base.d0) & (base.DM <= END)].copy()
inc["ap_before"] = inc.AP_any.notna() & (inc.AP_any < inc.DM)
inc["ap_90"] = inc.AP_any.notna() & (inc.AP_any + pd.Timedelta(days=90) < inc.DM)
prior = lambda c: inc[c].notna() & (inc[c] <= inc.DM)
inc["cvd_free"] = ~prior("mi") & ~prior("stroke")
inc["pc_bad"] = inc.pcan.notna() & (inc.pcan <= inc.DM + pd.Timedelta(days=365))
fu_end = inc[["death", "lost"]].min(axis=1).fillna(END).clip(upper=END)
inc["mace_after"] = inc.mace.notna() & (inc.mace > inc.DM) & (inc.mace <= fu_end)
inc["hf_after"] = inc.hf.notna() & (inc.hf > inc.DM) & (inc.hf <= fu_end) & ~prior("hf")
inc["fu"] = (fu_end - inc.DM).dt.days / 365.25
OUT = []
P = lambda t: (print(t), OUT.append(t))
P(f"incident diabetes after baseline: {len(inc):,}; CVD-free at onset {int(inc.cvd_free.sum()):,}")
for lab, g in [("AP any time before diabetes", inc.ap_before), ("AP >90 d before diabetes", inc.ap_90), ("no AP before diabetes", ~inc.ap_before)]:
    s = inc[g & inc.cvd_free & ~inc.pc_bad]
    P(f"{lab}: n={len(s):,}; MACE after onset {int(s.mace_after.sum())} ; HF {int(s.hf_after.sum())}; median FU {s.fu.median():.1f} y; crude MACE rate/1000PY {s.mace_after.sum()/s.fu.sum()*1000:.1f}")
open(os.path.join(RES, "H04_cvd_after_ppdm.txt"), "w").write("\n".join(OUT) + "\n")
