"""16: pre-attack NMR metabolomics in the main matched cohort.
Prepares cache/H03_nmr_matched.csv for 17_nmr_models.R. The NMR table is read from UKB_NMR_FILE (default: the RAP extract in UKB_DATA_DIR).

Baseline (instance 0) Nightingale biomarkers, fields 23400-23648, are kept when <20% missing in the matched cohort.
Each measure is log-transformed (non-positive values set to half the smallest positive value; missing values stay missing,
so that participants without NMR data are not assigned extreme values), centred within spectrometer (field 23650)
and scaled to SD units, then clipped at +-5 SD.
Also writes cholesterol-lowering medication at baseline (fields 6177/6153, code 1) and time since last meal (field 74)
for sensitivity analyses, and the etiology of the matched case."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

D = os.environ.get("UKB_DATA_DIR", "../ukb_data")
NMR = os.environ.get("UKB_NMR_FILE", f"{D}/UKB619657_NMR_metabolomics_with_eid_2025Nov.csv")
AC = f"{D}/UKB619657_assessment_center_2025Nov.csv"
mt = pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl"))
ids = set(mt.eid)
hdr = pd.read_csv(NMR, nrows=0).columns
mcols = [c for c in hdr if c.endswith("_i0") and c[1:6].isdigit() and 23400 <= int(c[1:6]) <= 23648]
nm = pd.concat([ch[ch.eid.isin(ids)] for ch in pd.read_csv(NMR, usecols=["eid", "p23650_i0"] + mcols, chunksize=50000)]).set_index("eid")
spec = nm.pop("p23650_i0")
X = nm.apply(pd.to_numeric, errors="coerce")
print(f"{len(mcols)} NMR measures; {len(X):,} matched participants; all-missing NMR {int(X.isna().all(axis=1).sum())}")
X = X[X.columns[X.isna().mean() < 0.2]]
for c in X.columns:
    v = X[c]; pos = v[v > 0]
    if len(pos) == 0: continue
    X[c] = np.log(v.where((v > 0) | v.isna(), pos.min() / 2))
sp = spec.reindex(X.index).fillna(-1).astype(int).astype(str)
X = X - X.groupby(sp).transform("mean")
X = ((X - X.mean()) / X.std()).clip(-5, 5)
X.columns = [c.replace("_i0", "") for c in X.columns]
print(f"{X.shape[1]} measures kept")

ac = pd.concat([ch[ch.eid.astype(int).isin(ids)] for ch in pd.read_csv(AC, usecols=["eid", "p6177_i0", "p6153_i0", "p74_i0"], dtype=str, chunksize=100000)])
ac["eid"] = ac.eid.astype(int)
med = (ac.p6177_i0.fillna("") + "|" + ac.p6153_i0.fillna("")).str.split("|")
ac["chol_med"] = med.map(lambda s: int("1" in s))
ac["fast_h"] = pd.to_numeric(ac.p74_i0, errors="coerce").where(lambda v: v >= 0)
m = mt.merge(ac[["eid", "chol_med", "fast_h"]], on="eid", how="left")
m["Y"] = (m.E == 1).astype(int); m["bmi5"] = m.bmi / 5; m["hba1c5"] = m.hba1c / 5
m["biliary"] = (m.mid.map(m[m.exp == 1].set_index("mid").etiol) == "biliary").astype(int)
m = m[["eid", "mid", "exp", "T", "Y", "bmi5", "hba1c5", "biliary", "chol_med", "fast_h"]].merge(X, left_on="eid", right_index=True, how="left")
m.to_csv(os.path.join(ROOT, "cache", "H03_nmr_matched.csv"), index=False)
has = m[X.columns[0]].notna()
print(f"rows {len(m):,}; with NMR {int(has.sum()):,} (AP {int((has & (m.exp == 1)).sum()):,}, events {int(m[has & (m.exp == 1)].Y.sum())}; "
      f"controls {int((has & (m.exp == 0)).sum()):,}, events {int(m[has & (m.exp == 0)].Y.sum())}); cholesterol-lowering drugs {m.chol_med.mean()*100:.1f}%")
pd.Series({c: c for c in X.columns}).to_csv(os.path.join(ROOT, "cache", "H03_nmr_measures.csv"), header=False)
