"""01: Build the analysis cohort (cache/cohort_full.pkl).

Reads the UK Biobank Research Analysis Platform (RAP) table extracts from UKB_DATA_DIR:
first-occurrence dates, hospital inpatient (HES) ICD-10/ICD-9/OPCS-4 summary arrays, cancer registry,
death, baseline biochemistry and baseline covariates.

health_outcomes is read in chunks (reading ~500 string columns at once exhausts memory); HES codes are paired
with their first-date arrays by position. Baseline date and covariates are taken from the raw fields
p53 (assessment date), p21001 (BMI), p20116 (smoking status), p20117 (alcohol drinker status), p21000 (ethnic
background) and p22189 (Townsend deprivation index).

Row order affects which controls are drawn during matching. Rows are sorted by eid unless PPDM_ROW_ORDER points to
a one-column CSV of eids giving the order to use (used to verify the published results).
"""
import os, time
import numpy as np, pandas as pd
from common import ROOT

t0 = time.time()
M = os.environ.get("UKB_DATA_DIR", "../ukb_data")
TAB = {k: os.path.join(M, f"UKB619657_{k}_2025Nov.csv") for k in ["health_outcomes", "population_characteristics", "biological_samples", "assessment_center"]}
OUT = os.path.join(ROOT, "cache", "cohort_full.pkl")
SPECIAL = {1900, 1901, 1902, 1903, 2037}          # UK Biobank placeholder years for unknown dates
FO = {"ap": "p131682", "e10": "p130706", "e11": "p130708", "e12": "p130710", "e13": "p130712", "e14": "p130714",
      "gs": "p131674", "chole": "p131676", "appx": "p131604", "k86": "p131684", "k70": "p131658", "cat": "p131164", "koa": "p131872"}
HES = {  # name: (code column, date-array prefix, code prefixes)
    "h_ap10": ("p41270", "p41280_a", ("K85",)), "h_ap9": ("p41271", "p41281_a", ("5770",)),
    "h_dm10": ("p41270", "p41280_a", ("E10", "E11", "E12", "E13", "E14")), "h_dm9": ("p41271", "p41281_a", ("250",)),
    "h_cp10": ("p41270", "p41280_a", ("K860", "K861")), "h_cp9": ("p41271", "p41281_a", ("5771",)),
    "h_pcan": ("p41270", "p41280_a", ("C25",)), "h_appx": ("p41270", "p41280_a", ("K35",)), "h_chole": ("p41270", "p41280_a", ("K81",)),
    "h_chx": ("p41272", "p41282_a", ("J18",)),
}
F = TAB["health_outcomes"]
cols = pd.read_csv(F, nrows=0).columns
dcols = {p: [c for c in cols if c.startswith(p)] for p in ["p41280_a", "p41281_a", "p41282_a"]}
can = [f"p40006_i{i}" for i in range(22)] + [f"p40005_i{i}" for i in range(22)]
use = ["eid", "p40000_i0", "p41270", "p41271", "p41272", "p131683"] + list(FO.values()) + can + sum(dcols.values(), [])


def pair(ch, codecol, dprefix):
    """Return DataFrame(row, code, date): the j-th code is paired with the j-th date."""
    ex = ch[codecol].fillna("").str.split("|").explode()
    ex = pd.DataFrame({"row": ex.index, "code": ex.values}); ex["pos"] = ex.groupby("row").cumcount()
    ex = ex[ex.code.notna() & (ex.code != "")]
    D = ch[dcols[dprefix]]; D.columns = range(D.shape[1])
    st = D.stack()  # (row, pos) -> date string; missing values dropped
    ex["date"] = st.reindex(pd.MultiIndex.from_arrays([ex.row.values, ex.pos.values])).values
    return ex


parts = []
for k, ch in enumerate(pd.read_csv(F, usecols=use, dtype=str, chunksize=25000)):
    ch = ch.reset_index(drop=True)
    o = pd.DataFrame({"eid": ch.eid.astype(int)})
    for name, c in FO.items():
        d = pd.to_datetime(ch[c], errors="coerce"); bad = d.dt.year.isin(SPECIAL)
        o[name] = d.mask(bad); o[name + "_bad"] = bad
    o["ap_src"] = ch.p131683
    o["death"] = pd.to_datetime(ch.p40000_i0, errors="coerce")
    pcd = pd.Series(pd.NaT, index=ch.index, dtype="datetime64[ns]")
    for i in range(22):
        m = ch[f"p40006_i{i}"].fillna("").str.startswith("C25"); d = pd.to_datetime(ch[f"p40005_i{i}"], errors="coerce")
        pcd = pcd.where(~(m & (pcd.isna() | (d < pcd))), d)
    o["pcan_reg"] = pcd
    cache = {}
    for name, (cc, dp, pre) in HES.items():
        if cc not in cache: cache[cc] = pair(ch, cc, dp)
        ex = cache[cc]; s = ex[ex.code.str.startswith(pre)]
        o[name] = pd.to_datetime(s.groupby("row").date.min(), errors="coerce").reindex(ch.index)
    ex = cache["p41270"]
    o["k851"] = o.index.isin(ex.row[ex.code == "K851"]); o["k852"] = o.index.isin(ex.row[ex.code == "K852"])
    o["f10"] = o.index.isin(ex.row[ex.code.str.startswith("F10")])
    parts.append(o)
    print(f"health_outcomes chunk {k+1}: {sum(len(p) for p in parts):,} rows, {round(time.time()-t0)} s", flush=True)
out = pd.concat(parts, ignore_index=True); del parts
out["h_ap"] = out[["h_ap10", "h_ap9"]].min(axis=1); out["h_dm"] = out[["h_dm10", "h_dm9"]].min(axis=1); out["h_cp"] = out[["h_cp10", "h_cp9"]].min(axis=1)
out["pcan"] = out[["pcan_reg", "h_pcan"]].min(axis=1)
dmc = out[["e10", "e11", "e12", "e13", "e14"]]
out["dm"] = dmc.min(axis=1); out["dm_bad"] = out[[f"e1{i}_bad" for i in range(5)]].any(axis=1)
has = dmc.notna().any(axis=1); out["dm_type"] = pd.Series(np.nan, index=out.index, dtype=object)
out.loc[has, "dm_type"] = dmc[has].idxmin(axis=1)  # idxmin fails on all-missing rows

# ---- baseline date and covariates from raw fields ----
ac = []
for ch in pd.read_csv(TAB["assessment_center"], usecols=["eid", "p53_i0", "p21000_i0", "p21001_i0", "p20116_i0", "p20117_i0"], dtype=str, chunksize=100000):
    ac.append(ch)
ac = pd.concat(ac, ignore_index=True); ac["eid"] = ac.eid.astype(int)
num = lambda s: pd.to_numeric(s, errors="coerce")
nonneg = lambda s: num(s).where(num(s) >= 0)          # negative codes = do not know / prefer not to answer
eth = num(ac.p21000_i0)
b = pd.DataFrame({"eid": ac.eid, "d0": pd.to_datetime(ac.p53_i0, errors="coerce")})
cov = pd.DataFrame({"eid": ac.eid, "bmi": num(ac.p21001_i0), "smoke": nonneg(ac.p20116_i0), "alcohol": nonneg(ac.p20117_i0),
                    "ethnic": np.where(eth.isin([1, 1001, 1002, 1003]), 1.0, np.where(eth > 0, 2.0, np.nan))})   # 1 = White, 2 = other
pc = pd.read_csv(TAB["population_characteristics"], usecols=["eid", "p31", "p34", "p52", "p191", "p22189"])
cov = cov.merge(pc[["eid", "p22189"]].rename(columns={"p22189": "tdi"}), on="eid", how="left")
order = os.environ.get("PPDM_ROW_ORDER")
if order:
    ids = pd.read_csv(order).iloc[:, 0].astype(int)
    b = b.set_index("eid").reindex(pd.Index(ids.values, name="eid")).dropna(how="all").reset_index()
else:
    b = b.sort_values("eid").reset_index(drop=True)
pc["lost"] = pd.to_datetime(pc.p191, errors="coerce")
pc["birth"] = pd.to_datetime(pc.p34.astype("Int64").astype(str) + "-" + pc.p52.fillna(7).astype(int).astype(str) + "-15", errors="coerce")
bs = pd.read_csv(TAB["biological_samples"], usecols=["eid", "p30750_i0", "p30740_i0", "p30870_i0"]).rename(columns={"p30750_i0": "hba1c", "p30740_i0": "glucose", "p30870_i0": "tg"})
df = b.merge(pc[["eid", "p31", "p34", "birth", "lost"]], on="eid").merge(out, on="eid").merge(bs, on="eid", how="left").merge(cov, on="eid", how="left")
df.to_pickle(OUT); print("saved", df.shape, round(time.time() - t0), "s")
print("check: AP (first occurrence)", df.ap.notna().sum(), "| AP (HES)", df.h_ap.notna().sum(), "| diabetes", df.dm.notna().sum(),
      "| chronic pancreatitis (HES)", df.h_cp.notna().sum(), "| pancreatic cancer", df.pcan.notna().sum(), "| cholecystectomy", df.h_chx.notna().sum())
