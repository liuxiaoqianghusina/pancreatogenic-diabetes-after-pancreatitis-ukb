"""02: Main analysis. AP vs. population controls matched on pre-attack HbA1c and BMI; effect modification on the
multiplicative and additive scales; subgroups; active comparators (acute appendicitis, acute cholecystitis without AP).
Usage: python 02_main.py [n_bootstrap=300]"""
import sys, numpy as np, pandas as pd
from common import *

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
df = load()
base = base_cohort(df)
cases = exposed(base, "AP"); cases["etiol"] = etiology(cases)
m = covars(follow(match(base, cases)))
m["age_t0"] = (m.t0 - m.birth).dt.days / 365.25
m.to_pickle(os.path.join(ROOT, "cache", "matched_main.pkl"))
log = open(os.path.join(RES, "F02_summary.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

P(f"Eligible at baseline {len(base):,}; AP cases {len(cases):,}; matched AP {int((m.exp==1).sum()):,}, controls {int((m.exp==0).sum()):,}")
P("Etiology:", cases.etiol.value_counts().to_dict())
gap = (cases.t_evt - cases.d0).dt.days / 365.25
P(f"Baseline HbA1c to AP: median {gap.median():.1f} years (IQR {gap.quantile(.25):.1f}–{gap.quantile(.75):.1f})")

# ---- characteristics at the index date ----
def desc(d):
    return {"n": len(d), "Age at index date": f"{d.age_t0.mean():.1f} ± {d.age_t0.std():.1f}", "Women %": f"{(d.male==0).mean()*100:.1f}",
            "HbA1c mmol/mol": f"{d.hba1c.median():.1f} ({d.hba1c.quantile(.25):.1f}–{d.hba1c.quantile(.75):.1f})",
            "BMI": f"{d.bmi.mean():.1f} ± {d.bmi.std():.1f}", "Current smoking %": f"{d.smk_cur.mean()*100:.1f}", "Current drinking %": f"{d.alc_cur.mean()*100:.1f}",
            "Townsend": f"{d.tdi.mean():.2f} ± {d.tdi.std():.2f}", "White %": f"{d.white.mean()*100:.1f}",
            "Triglycerides mmol/L": f"{d.tg.median():.2f} ({d.tg.quantile(.25):.2f}–{d.tg.quantile(.75):.2f})",
            "Cholelithiasis before index %": f"{d.gs_prior.mean()*100:.1f}", "Median follow-up, years": f"{d['T'].median():.1f} ({d['T'].quantile(.25):.1f}–{d['T'].quantile(.75):.1f})"}
t1 = pd.DataFrame({"AP": desc(m[m.exp == 1]), "Matched controls": desc(m[m.exp == 0])}); P(t1.to_string())
t1.to_csv(os.path.join(RES, "F02_table1.csv"), encoding="utf-8-sig")

# ---- main results: overall and by pre-attack HbA1c ----
rows = []
def main_row(lab, d):
    a, c = d[d.exp == 1], d[d.exp == 0]
    r1 = cox(d); r2 = cox(d, covs=ADJ_FULL + ["gs_prior"])
    ca, cc = cif(a, var=True), cif(c, var=True)
    return dict(group=lab, AP=f"{r1['n1']}/{r1['e1']}", controls=f"{r1['n0']}/{r1['e0']}", IR_AP=round(rate(a), 1), IR_ctrl=round(rate(c), 1),
                HR_matched=fmt(r1), P=f"{r1['p']:.1e}", HR_adj=fmt(r2), CIF5_AP=f"{ca[0]:.1f} ({ca[1]:.1f}–{ca[2]:.1f})",
                CIF5_ctrl=f"{cc[0]:.1f} ({cc[1]:.1f}–{cc[2]:.1f})", RD5=round(ca[0] - cc[0], 1), _hr=r1["HR"])
rows.append(main_row("All", m))
for h in HLAB: rows.append(main_row(f"HbA1c {h}", m[m.hcat == h]))
# multiplicative interaction: exp × continuous HbA1c (per 5 mmol/mol) in the matched-set-stratified model
mm = m.copy(); mm["xh"] = mm.exp * (mm.hba1c - 36) / 5
f = CoxPHFitter().fit(mm[["T", "E", "exp", "xh", "mid"]].assign(E=(mm.E == 1).astype(int)), "T", "E", strata=["mid"])
p_int = f.summary.loc["xh", "p"]; hr_int = f.summary.loc["xh", "exp(coef)"]
P(f"Multiplicative interaction AP × HbA1c (per 5 mmol/mol): HR ratio {hr_int:.2f}, P = {p_int:.4f}")
# trend across HbA1c categories (ordinal)
mm["xo"] = mm.exp * mm.hcat.cat.codes
f2 = CoxPHFitter().fit(mm[["T", "E", "exp", "xo", "mid"]].assign(E=(mm.E == 1).astype(int)), "T", "E", strata=["mid"])
P(f"Multiplicative interaction (ordinal HbA1c category) P = {f2.summary.loc['xo','p']:.4f}")

# ---- additive interaction (RERI) and bootstrap of risk differences (resampling matched sets) ----
def reri(d, cut):
    d = d.copy(); d["pre"] = (d.hba1c >= cut).astype(int); d["xp"] = d.exp * d.pre; d["EE"] = (d.E == 1).astype(int)
    X = d[["T", "EE", "exp", "pre", "xp", "age_t0", "male", "bmi"]]
    try: f = CoxPHFitter().fit(X, "T", "EE")
    except Exception: f = CoxPHFitter().fit(X, "T", "EE", fit_options={"step_size": 0.3})  # smaller Newton steps if the default fails
    b = f.params_
    h10, h01, h11 = np.exp(b["exp"]), np.exp(b["pre"]), np.exp(b["exp"] + b["pre"] + b["xp"])
    return h11 - h10 - h01 + 1, h10, h01, h11
def rds(d):
    out = []
    for h in HLAB:
        s = d[d.hcat == h]; out.append(cif(s[s.exp == 1]) - cif(s[s.exp == 0]))
    return out
est39, est42 = reri(m, 39), reri(m, 42); est_rd = rds(m)
rng = np.random.default_rng(1); mids = m.mid.unique(); g = {k: v for k, v in m.groupby("mid")}
bt = []
for b in range(B):
    s = rng.choice(mids, len(mids), replace=True)
    d = pd.concat([g[k].assign(mid=i) for i, k in enumerate(s)], ignore_index=True)
    try: bt.append([reri(d, 39)[0], reri(d, 42)[0]] + rds(d))
    except Exception: pass
bt = np.array(bt)
ci = lambda j: (np.percentile(bt[:, j], 2.5), np.percentile(bt[:, j], 97.5))
P(f"bootstrap samples {len(bt)}")
P(f"RERI (ADA prediabetes, HbA1c ≥39 vs <39) {est39[0]:.2f} ({ci(0)[0]:.2f}–{ci(0)[1]:.2f}); HR10 {est39[1]:.2f} HR01 {est39[2]:.2f} HR11 {est39[3]:.2f}")
P(f"RERI (NICE prediabetes, HbA1c ≥42 vs <42) {est42[0]:.2f} ({ci(1)[0]:.2f}–{ci(1)[1]:.2f}); HR10 {est42[1]:.2f} HR01 {est42[2]:.2f} HR11 {est42[3]:.2f}")
for j, h in enumerate(HLAB):
    lo, hi = ci(2 + j); rows[1 + j]["RD5_CI"] = f"{est_rd[j]:.1f} ({lo:.1f}–{hi:.1f})"
R = pd.DataFrame(rows).drop(columns="_hr"); P(R.to_string(index=False))
R.to_csv(os.path.join(RES, "F02_main.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([dict(measure="Multiplicative interaction, HbA1c per 5", value=f"{hr_int:.2f}", P=f"{p_int:.4f}"),
              dict(measure="RERI ≥39", value=f"{est39[0]:.2f} ({ci(0)[0]:.2f}–{ci(0)[1]:.2f})", P=""),
              dict(measure="RERI ≥42", value=f"{est42[0]:.2f} ({ci(1)[0]:.2f}–{ci(1)[1]:.2f})", P="")]).to_csv(os.path.join(RES, "F02_interaction.csv"), index=False, encoding="utf-8-sig")

# ---- subgroups: etiology, follow-up period, sex, age, BMI ----
sub = []
def srow(lab, d):
    r = cox(d); sub.append(dict(subgroup=lab, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), P=f"{r['p']:.1e}"))
for e in ["biliary", "alcohol", "other"]: srow(f"Etiology {e}", m[m.etiol == e])
for a, b_, lab in [(0, 1, "0–1 year after index"), (1, 3, "1–3 years"), (3, 99, ">3 years")]:
    d = m[m["T"] > a].copy(); d["E"] = np.where(d["T"] <= b_, d.E, 0); d["T"] = d["T"].clip(upper=b_) - a; srow(f"Follow-up {lab}", d[d["T"] > 0])
srow("Men", m[m.male == 1]); srow("Women", m[m.male == 0])
srow("Age at AP <65", m[m.case_eid.map(cases.set_index("eid").age_evt) < 65]); srow("Age at AP ≥65", m[m.case_eid.map(cases.set_index("eid").age_evt) >= 65])
for bl in BLAB: srow(f"BMI {bl}", m[m.bcat == bl])
S = pd.DataFrame(sub); P(S.to_string(index=False)); S.to_csv(os.path.join(RES, "F02_subgroups.csv"), index=False, encoding="utf-8-sig")

# ---- type of the first diabetes code ----
ev = m[m.E == 1]
P("First diabetes code (%), AP:", (ev[ev.exp == 1].dm_type.value_counts(normalize=True) * 100).round(1).to_dict())
P("First diabetes code (%), controls:", (ev[ev.exp == 0].dm_type.value_counts(normalize=True) * 100).round(1).to_dict())

# ---- active comparators: acute appendicitis and acute cholecystitis without AP ----
comp = {"appendicitis": exposed(base, "appx"), "cholecystitis": exposed(base, "chole")}
A = [follow(cases.assign(exp=1, grp="AP"), censor_ap_ctrl=False)]
for k, e in comp.items():
    A.append(follow(e.assign(exp=0, grp=k), censor_ap_ctrl=False))  # AP within 1 year of the comparator event was excluded in exposed()
A = pd.concat(A, ignore_index=True)
# comparator patients are censored at a later AP
c = (A.grp != "AP") & A.AP_any.notna() & (A.AP_any > A.t0) & (A.AP_any < A.t0 + pd.to_timedelta(A["T"] * 365.25, unit="D"))
A.loc[c, "E"] = 0; A.loc[c, "T"] = (A.loc[c, "AP_any"] - A.loc[c, "t0"]).dt.days / 365.25
A = covars(A[A["T"] > 0]); A["x"] = (A.grp == "AP").astype(int)
A["chx"] = (A.h_chx.notna() & (A.h_chx <= A.t_evt + pd.Timedelta(days=180))).astype(int)
A["etiol"] = np.where(A.grp == "AP", A.eid.map(cases.set_index("eid").etiol), "")
act = []
def arow(lab, d, covs):
    r = cox(d, x="x", covs=covs, strata=None)
    act.append(dict(comparison=lab, AP=f"{r['n1']}/{r['e1']}", comparator=f"{r['n0']}/{r['e0']}", HR=fmt(r), P=f"{r['p']:.1e}"))
base_c = ["age_evt", "male"]
for k in comp:
    d = A[A.grp.isin(["AP", k])]
    arow(f"vs {k} | age and sex", d, base_c); arow(f"vs {k} | + HbA1c and BMI", d, base_c + ["hba1c", "bmi"]); arow(f"vs {k} | fully adjusted", d, base_c + ADJ_FULL)
arow("vs both | fully adjusted", A, base_c + ADJ_FULL)
for h in HLAB: arow(f"vs both | HbA1c {h} | fully adjusted", A[A.hcat == h], base_c + ADJ_FULL)
d = A[(A.grp == "cholecystitis") | ((A.grp == "AP") & (A.etiol == "biliary"))]
arow("biliary AP vs cholecystitis | fully adjusted", d, base_c + ADJ_FULL); arow("biliary AP vs cholecystitis | fully adjusted + cholecystectomy", d, base_c + ADJ_FULL + ["chx"])
P("Cholecystectomy by 180 days after the event (%): biliary AP", round(d[d.x == 1].chx.mean() * 100, 1), "cholecystitis", round(d[d.x == 0].chx.mean() * 100, 1))
AC = pd.DataFrame(act); P(AC.to_string(index=False)); AC.to_csv(os.path.join(RES, "F02_active.csv"), index=False, encoding="utf-8-sig")
P("5-year cumulative incidence (%), active-comparator cohorts:", {k: round(cif(A[A.grp == k]), 1) for k in ["AP", "appendicitis", "cholecystitis"]})
log.close()
