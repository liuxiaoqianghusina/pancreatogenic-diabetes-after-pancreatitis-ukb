"""01b: First HES dates of organ failure (N17 acute kidney injury, J96 respiratory failure, R57 shock, R65 SIRS) and pancreatic
pseudocyst (K86.3), used as proxies of AP severity. Note: p41280 gives only the first date of each code, so a code first recorded
before the AP cannot be detected again at the AP admission (conservative misclassification as non-severe)."""
import pandas as pd, os, time
t0 = time.time()
M = os.environ.get("UKB_DATA_DIR", "../ukb_data"); F = f"{M}/UKB619657_health_outcomes_2025Nov.csv"
from common import ROOT
OUT = os.path.join(ROOT, "cache", "extra_codes.pkl")
cols = pd.read_csv(F, nrows=0).columns; dc = [c for c in cols if c.startswith("p41280_a")]
CODES = {"h_aki": ("N17",), "h_resp": ("J96",), "h_shock": ("R57",), "h_sirs": ("R65",), "h_pseudo": ("K863",)}
parts = []
for ch in pd.read_csv(F, usecols=["eid", "p41270"] + dc, dtype=str, chunksize=25000):
    ch = ch.reset_index(drop=True)
    ex = ch.p41270.fillna("").str.split("|").explode(); ex = pd.DataFrame({"row": ex.index, "code": ex.values}); ex["pos"] = ex.groupby("row").cumcount()
    ex = ex[ex.code.notna() & (ex.code != "")]
    D = ch[dc]; D.columns = range(D.shape[1]); st = D.stack()
    ex["date"] = st.reindex(pd.MultiIndex.from_arrays([ex.row.values, ex.pos.values])).values
    o = pd.DataFrame({"eid": ch.eid.astype(int)})
    for k, pre in CODES.items():
        s = ex[ex.code.str.startswith(pre)]; o[k] = pd.to_datetime(s.groupby("row").date.min(), errors="coerce").reindex(ch.index)
    parts.append(o)
pd.concat(parts, ignore_index=True).to_pickle(OUT); print("done", round(time.time() - t0), "s")
