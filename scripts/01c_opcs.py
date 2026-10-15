"""01c: First dates of pancreas-related OPCS-4 procedures (p41272 codes paired with p41282 dates by position).
- necrosectomy J57.6; drainage of pancreatic lesion: open J61, percutaneous J66, endoscopic J42.4, pancreatic duct drainage J60.1
- pancreatic resection (iatrogenic diabetes risk): J55, J56, J57 (excluding J57.6), J58
- therapeutic ERCP (biliary): J38–J41"""
import pandas as pd, os, time
t0 = time.time()
M = os.environ.get("UKB_DATA_DIR", "../ukb_data"); F = f"{M}/UKB619657_health_outcomes_2025Nov.csv"
from common import ROOT
OUT = os.path.join(ROOT, "cache", "opcs_pancreas.pkl")
cols = pd.read_csv(F, nrows=0).columns; dc = [c for c in cols if c.startswith("p41282_a")]
G = {"op_necro": ("J576",), "op_drain": ("J61", "J66", "J424", "J601"),
     "op_resect": ("J55", "J56", "J571", "J572", "J573", "J574", "J575", "J578", "J579", "J58"), "op_ercp": ("J38", "J39", "J40", "J41")}
parts = []
for ch in pd.read_csv(F, usecols=["eid", "p41272"] + dc, dtype=str, chunksize=25000):
    ch = ch.reset_index(drop=True)
    ex = ch.p41272.fillna("").str.split("|").explode(); ex = pd.DataFrame({"row": ex.index, "code": ex.values}); ex["pos"] = ex.groupby("row").cumcount()
    ex = ex[ex.code.notna() & (ex.code != "")]; ex["code"] = ex.code.str.split(" ").str[0].str.replace(".", "", regex=False)
    D = ch[dc]; D.columns = range(D.shape[1]); st = D.stack()
    ex["date"] = st.reindex(pd.MultiIndex.from_arrays([ex.row.values, ex.pos.values])).values
    o = pd.DataFrame({"eid": ch.eid.astype(int)})
    for k, pre in G.items():
        s = ex[ex.code.str.startswith(pre)]; o[k] = pd.to_datetime(s.groupby("row").date.min(), errors="coerce").reindex(ch.index)
    parts.append(o)
out = pd.concat(parts, ignore_index=True); out.to_pickle(OUT)
print("done", round(time.time() - t0), "s; n:", {k: int(out[k].notna().sum()) for k in G})
