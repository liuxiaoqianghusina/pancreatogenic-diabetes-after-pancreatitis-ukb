"""11: Type 2 diabetes polygenic risk score (PRS; PRScs, z score) triangulation.
1) PRS validity: HR per SD of PRS for incident diabetes in the full baseline cohort (positive control).
2) Whether AP patients carry a higher genetic predisposition than participants of the same age and sex (standardized difference).
3) Main analysis: exact matching on PRS tertile in addition to the original matching factors (sex, birth year, HbA1c category,
   BMI category); HRs, 5-year risk differences and attributable fractions (bootstrap) by PRS tertile; multiplicative interaction
   (AP × continuous PRS) and additive interaction (RERI, highest vs lowest tertile).
4) Pre-attack glycemia × genetic risk: the AP effect in people with HbA1c <39 mmol/mol and PRS in the lowest tertile.
5) The original matched cohort with additional adjustment for PRS and 10 genetic principal components.

The PRS file (UKB_PRS_FILE) needs columns eid, PRScs_z, Batch and pc1–pc10.
Usage: python 11_prs.py [bootstrap replicates, default 300]. Requires cache/matched_main.pkl from 02_main.py."""
import sys, numpy as np, pandas as pd
from lifelines import CoxPHFitter
from common import *

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PRS = os.environ.get("UKB_PRS_FILE", "../ukb_data/T2D_PRScs_zscore.csv")
out = open(os.path.join(RES, "H01_summary.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")

prs = pd.read_csv(PRS, usecols=["eid", "PRScs_z", "Batch"] + [f"pc{i}" for i in range(1, 11)])
prs = prs.dropna(subset=["PRScs_z"]).drop_duplicates("eid")
df = load().merge(prs, on="eid", how="left")
base = base_cohort(df); g = base[base.PRScs_z.notna()].copy()
g["prs"] = (g.PRScs_z - g.PRScs_z.mean()) / g.PRScs_z.std()
cut = g.prs.quantile([1 / 3, 2 / 3]).values
base["prs"] = (base.PRScs_z - g.PRScs_z.mean()) / g.PRScs_z.std()
T = ["T1 low", "T2 intermediate", "T3 high"]
base["prs_t"] = pd.cut(base.prs, [-np.inf, cut[0], cut[1], np.inf], labels=T)
P(f"PRS file {len(prs):,} participants; baseline cohort {len(base):,}, with PRS {base.prs.notna().sum():,} ({base.prs.notna().mean()*100:.1f}%)")

# ---------- 1. positive control: PRS → incident diabetes in the full cohort ----------
q = base[base.prs.notna()].copy(); q["t0"] = q.d0; q["exp"] = 0
q = follow(q, censor_ap_ctrl=False); q["age0"] = (q.d0 - q.birth).dt.days / 365.25
X = q[["T", "E", "prs", "age0", "male"] + [f"pc{i}" for i in range(1, 11)]].copy(); X["E"] = (X.E == 1).astype(int)
f = CoxPHFitter().fit(X, "T", "E"); s = f.summary.loc["prs"]
P(f"[1] PRS per SD and incident diabetes (full cohort n={len(X):,}, events {int(X.E.sum()):,}): HR {s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}–{s['exp(coef) upper 95%']:.2f})")
X2 = q[["T", "E", "prs", "age0", "male", "hba1c", "bmi"] + [f"pc{i}" for i in range(1, 11)]].copy(); X2["E"] = (X2.E == 1).astype(int)
s2 = CoxPHFitter().fit(X2, "T", "E").summary.loc["prs"]
P(f"    further adjusted for HbA1c and BMI: HR {s2['exp(coef)']:.2f} ({s2['exp(coef) lower 95%']:.2f}–{s2['exp(coef) upper 95%']:.2f})")
c3 = q.groupby("prs_t", observed=True).apply(lambda d: cif(d.assign(t0=d.d0), t=10)).round(1).to_dict()
P("    10-year cumulative incidence of diabetes by tertile (%):", c3)

# ---------- 2. PRS of AP patients vs participants of the same age and sex ----------
cases_all = exposed(base, "AP"); cg = cases_all[cases_all.prs.notna()]
rng = np.random.default_rng(3); keep = ["eid", "p31", "p34", "d0", "DM", "dm", "death", "lost", "AP_any", "prs"]
grp = {k: v for k, v in base[base.prs.notna()][keep].groupby(["p31", "p34"])}; ctl = []
for _, c in cg.iterrows():
    gg = grp[(c.p31, c.p34)]; t = c.t0
    ok = gg[(gg.eid != c.eid) & (gg.d0 < c.t_evt) & ~(gg.DM.notna() & (gg.DM <= t)) & ~(gg.death.notna() & (gg.death <= t)) & ~(gg.AP_any.notna() & (gg.AP_any <= t))]
    ctl.append(ok.iloc[rng.choice(len(ok), min(5, len(ok)), replace=False)])
ctl = pd.concat(ctl); smd = (cg.prs.mean() - ctl.prs.mean()) / np.sqrt((cg.prs.var() + ctl.prs.var()) / 2)
from scipy.stats import mannwhitneyu
P(f"[2] {len(cg):,} AP patients have a PRS ({len(cg)/len(cases_all)*100:.1f}%). Mean PRS AP {cg.prs.mean():.3f} vs same age and sex {ctl.prs.mean():.3f}; SMD {smd:.3f}; P = {mannwhitneyu(cg.prs, ctl.prs).pvalue:.3f}")
P("    PRS tertile distribution of AP patients (%):", (cg.prs_t.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())

# ---------- 3. additional matching on PRS tertile ----------
def match_keys(base, cases, keys, k=5, seed=20261015):
    rng = np.random.default_rng(seed)
    kp = ["eid", "d0", "DM", "dm", "death", "lost", "AP_any", "pcan", "h_cp"] + keys
    grp = {kk: gg for kk, gg in base[kp].groupby(keys, observed=True)}; recs = []
    for mid, (_, c) in enumerate(cases.iterrows()):
        gg = grp.get(tuple(c[x] for x in keys)); t = c.t0
        if gg is None: continue
        ok = gg[(gg.eid != c.eid) & (gg.d0 < c.t_evt) & ~(gg.DM.notna() & (gg.DM <= t)) & ~(gg.dm.notna() & (gg.dm <= t)) & ~(gg.death.notna() & (gg.death <= t))
                & ~(gg.lost.notna() & (gg.lost <= t)) & ~(gg.AP_any.notna() & (gg.AP_any <= t)) & ~(gg.pcan.notna() & (gg.pcan <= t)) & ~(gg.h_cp.notna() & (gg.h_cp <= t))]
        if len(ok) == 0: continue
        sel = ok.eid.values[rng.choice(len(ok), min(k, len(ok)), replace=False)]
        recs.append((mid, 1, c.eid, t, c.t_evt)); recs += [(mid, 0, s_, t, c.t_evt) for s_ in sel]
    m = pd.DataFrame(recs, columns=["mid", "exp", "eid", "t0", "t_evt"]).merge(base.drop(columns=[c for c in ["t0", "t_evt"] if c in base]), on="eid")
    return follow(m)
bp = base[base.prs.notna()]
m = match_keys(bp, cg, ["p31", "p34", "hcat", "bcat", "prs_t"])
m["age_t0"] = (m.t0 - m.birth).dt.days / 365.25
m.to_pickle(os.path.join(ROOT, "cache", "matched_prs.pkl"))
P(f"[3] With PRS added to matching: AP {int((m.exp==1).sum()):,}, controls {int((m.exp==0).sum()):,}")
rows = []
def strata_stats(x):
    r = {}
    for t in ["All"] + T:
        s_ = x if t == "All" else x[x.prs_t == t]; a, c = cif(s_[s_.exp == 1]), cif(s_[s_.exp == 0]); r[t] = (a, c, a - c, (a - c) / a * 100 if a > 0 else np.nan)
    return r
def reri(d):
    d = d.copy(); d["hi"] = (d.prs_t == T[2]).astype(int); d = d[d.prs_t.isin([T[0], T[2]])]
    d["xh"] = d.exp * d.hi; d["EE"] = (d.E == 1).astype(int); Xr = d[["T", "EE", "exp", "hi", "xh", "age_t0", "male", "bmi", "hba1c"]]
    try: f = CoxPHFitter().fit(Xr, "T", "EE")
    except Exception: f = CoxPHFitter().fit(Xr, "T", "EE", fit_options={"step_size": 0.3})
    b = f.params_; h10, h01, h11 = np.exp(b["exp"]), np.exp(b["hi"]), np.exp(b["exp"] + b["hi"] + b["xh"]); return h11 - h10 - h01 + 1, h10, h01, h11
est = strata_stats(m); er = reri(m); mids = m.mid.unique(); gd = {k: v for k, v in m.groupby("mid")}; rg = np.random.default_rng(4); bt = []
for b in range(B):
    s_ = rg.choice(mids, len(mids), replace=True); x = pd.concat([gd[k].assign(mid=i) for i, k in enumerate(s_)], ignore_index=True)
    try: bt.append((strata_stats(x), reri(x)[0]))
    except Exception: pass
for t in ["All"] + T:
    s_ = m if t == "All" else m[m.prs_t == t]; r = cox(s_)
    rd = [b_[0][t][2] for b_ in bt]; af = [b_[0][t][3] for b_ in bt]
    rows.append(dict(PRS_tertile=t, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", incidence=f"{rate(s_[s_.exp==1]):.1f} vs {rate(s_[s_.exp==0]):.1f}",
                     HR=fmt(r), P=f"{r['p']:.1e}", CIF5_AP=round(est[t][0], 1), CIF5_ctrl=round(est[t][1], 1),
                     risk_difference=f"{est[t][2]:.1f} ({np.percentile(rd,2.5):.1f}–{np.percentile(rd,97.5):.1f})",
                     AF_exposed=f"{est[t][3]:.0f}% ({np.percentile(af,2.5):.0f}–{np.percentile(af,97.5):.0f})"))
R = pd.DataFrame(rows); P(R.to_string(index=False)); R.to_csv(os.path.join(RES, "H01_prs_strata.csv"), index=False, encoding="utf-8-sig")
rr = [b_[1] for b_ in bt]
P(f"    RERI (PRS T3 vs T1) {er[0]:.2f} ({np.percentile(rr,2.5):.2f}–{np.percentile(rr,97.5):.2f}); HR10 {er[1]:.2f} HR01 {er[2]:.2f} HR11 {er[3]:.2f}; {len(bt)} bootstrap replicates")
Xi = m[["T", "E", "exp", "prs", "mid"]].copy(); Xi["E"] = (Xi.E == 1).astype(int); Xi["xp"] = Xi.exp * Xi.prs
fi = CoxPHFitter().fit(Xi, "T", "E", strata=["mid"]); si = fi.summary.loc["xp"]
P(f"    Multiplicative interaction: per 1 SD higher PRS the HR for AP is multiplied by {si['exp(coef)']:.2f} ({si['exp(coef) lower 95%']:.2f}–{si['exp(coef) upper 95%']:.2f}), P = {si['p']:.4f}")

# ---------- 4. pre-attack glycemia × genetic risk ----------
jr = []
for hl, hf in [("HbA1c <39", m.hba1c < 39), ("HbA1c 39–47", m.hba1c >= 39)]:
    for t in T:
        s_ = m[hf & (m.prs_t == t)]
        keepm = s_.groupby("mid").exp.transform(lambda z: (z == 1).any() and (z == 0).any()); s_ = s_[keepm]
        if s_.exp.sum() < 10: continue
        r = cox(s_); jr.append(dict(HbA1c=hl, PRS=t, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r),
                                    CIF5=f"{cif(s_[s_.exp==1]):.1f} vs {cif(s_[s_.exp==0]):.1f}", risk_difference=round(cif(s_[s_.exp==1]) - cif(s_[s_.exp==0]), 1)))
J = pd.DataFrame(jr); P("[4] Pre-attack HbA1c × PRS"); P(J.to_string(index=False)); J.to_csv(os.path.join(RES, "H01_joint_hba1c_prs.csv"), index=False, encoding="utf-8-sig")

# ---------- 5. original matched cohort + adjustment for PRS and principal components ----------
m0 = pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl")).merge(prs, on="eid", how="left")
m0["prs"] = (m0.PRScs_z - g.PRScs_z.mean()) / g.PRScs_z.std()
keepm = m0.groupby("mid").apply(lambda d: d[d.exp == 1].prs.notna().all() and d[d.exp == 0].prs.notna().any())
m0 = m0[m0.mid.isin(keepm[keepm].index) & m0.prs.notna()]
pcs = [f"pc{i}" for i in range(1, 11)]
r_a, r_b = cox(m0), cox(m0, covs=["prs"] + pcs)
P(f"[5] Original matched cohort (participants with PRS, AP {r_a['n1']}): matched HR {fmt(r_a)} → + PRS + 10 principal components {fmt(r_b)}")
out.close()
