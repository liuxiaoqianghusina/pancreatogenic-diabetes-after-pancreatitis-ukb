"""06: Additional analyses — flow-chart counts, stratum summaries with attributable fractions among the exposed (bootstrap CIs),
1/3/5/10-year absolute risks, time to diagnosis,
appendicitis/cholecystitis versus their own matched populations, proportional-hazards test, E-values, covariate missingness,
and pre-attack HbA1c compared with age- and sex-matched participants.

Effect modification by continuous HbA1c is estimated in 06b_continuous.py.
Usage: python 06_additional.py [bootstrap replicates, default 300]. Requires cache/matched_main.pkl from 02_main.py."""
import sys, numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from common import *

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
out = open(os.path.join(RES, "F06_summary.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")

df = load()
# ---------- A. flow chart ----------
d = df.copy(); any_ap = d[["ap", "h_ap"]].min(axis=1); steps = [("UK Biobank participants", len(d))]
def step(lab, keep):
    global d, any_ap
    d = d[keep]; any_ap = any_ap[keep]; steps.append((lab, len(d)))
step("Excluded: diabetes at baseline (including unknown date)", (~d.dm_bad) & ~(d.dm.notna() & (d.dm <= d.d0)) & ~(d.h_dm.notna() & (d.h_dm <= d.d0)))
step("Excluded: baseline HbA1c missing", d.hba1c.notna())
step("Excluded: baseline HbA1c ≥48 mmol/mol (undiagnosed diabetes)", d.hba1c < 48)
step("Excluded: BMI missing", d.bmi.notna())
step("Excluded: acute pancreatitis before baseline (including unknown date)", ~d.ap_bad & ~(any_ap.notna() & (any_ap <= d.d0)))
step("Excluded: chronic pancreatitis before baseline", ~(d.h_cp.notna() & (d.h_cp <= d.d0)))
step("Excluded: pancreatic cancer before baseline", ~(d.pcan.notna() & (d.pcan <= d.d0)))
base = base_cohort(df); assert len(base) == steps[-1][1]
inc = base[base.AP.notna() & (base.AP > base.d0) & (base.AP <= END)]; steps.append(("First acute pancreatitis after baseline (to 2022-12-31)", len(inc)))
e = inc.copy(); e["t0"] = e.AP + pd.Timedelta(days=90)
for lab, bad in [("Excluded: diabetes before AP (incident after baseline) or within 90 days after AP", (e.DM.notna() & (e.DM <= e.t0)) | (e.dm.notna() & (e.dm <= e.t0))),
                 ("Excluded: death or loss to follow-up within 90 days after AP", (e.death.notna() & (e.death <= e.t0)) | (e.lost.notna() & (e.lost <= e.t0))),
                 ("Excluded: pancreatic cancer or chronic pancreatitis before or within 90 days after AP", (e.pcan.notna() & (e.pcan <= e.t0)) | (e.h_cp.notna() & (e.h_cp <= e.t0))),
                 ("Excluded: index date after end of follow-up", e.t0 >= END)]:
    e = e[~bad]; steps.append((lab, len(e)))
cases = exposed(base, "AP"); assert len(cases) == len(e)
m = pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl"))
steps.append(("Matched AP patients / matched controls", f"{int((m.exp==1).sum())} / {int((m.exp==0).sum())}"))
FL = pd.DataFrame(steps, columns=["step", "n"]); P(FL.to_string(index=False)); FL.to_csv(os.path.join(RES, "F06_flow.csv"), index=False, encoding="utf-8-sig")
P("Controls per case (number of controls: number of cases):", m[m.exp == 0].groupby("mid").size().value_counts().sort_index().to_dict())

# ---------- B. stratum summaries + attributable fractions (bootstrap) ----------
groups = {"All": lambda x: x.hba1c > 0, "HbA1c <36": lambda x: x.hcat == "<36", "HbA1c 36-38": lambda x: x.hcat == "36-38",
          "HbA1c 39-41": lambda x: x.hcat == "39-41", "HbA1c 42-47": lambda x: x.hcat == "42-47",
          "ADA normal <39": lambda x: x.hba1c < 39, "ADA prediabetes 39-47": lambda x: x.hba1c >= 39,
          "NICE normal <42": lambda x: x.hba1c < 42, "NICE prediabetes 42-47": lambda x: x.hba1c >= 42}
def stats(x):
    r = {}
    for k, f in groups.items():
        s = x[f(x)]; a, c = cif(s[s.exp == 1]), cif(s[s.exp == 0]); r[k] = (a, c, a - c, (a - c) / a * 100 if a > 0 else np.nan)
    return r
est = stats(m); rng = np.random.default_rng(2); mids = m.mid.unique(); g = {k: v for k, v in m.groupby("mid")}; bt = []
for b in range(B):
    s = rng.choice(mids, len(mids), replace=True); x = pd.concat([g[k].assign(mid=i) for i, k in enumerate(s)], ignore_index=True)
    bt.append(stats(x))
rows = []
for k, f in groups.items():
    s = m[f(m)]; r = cox(s)
    rd = [b[k][2] for b in bt]; af = [b[k][3] for b in bt]
    rows.append(dict(group=k, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), CIF5_AP=round(est[k][0], 1), CIF5_ctrl=round(est[k][1], 1),
                     risk_difference=f"{est[k][2]:.1f} ({np.percentile(rd,2.5):.1f}–{np.percentile(rd,97.5):.1f})",
                     AF_exposed=f"{est[k][3]:.0f}% ({np.percentile(af,2.5):.0f}–{np.percentile(af,97.5):.0f})",
                     NNS_5y=round(100 / est[k][0], 1)))
R = pd.DataFrame(rows); P(R.to_string(index=False)); R.to_csv(os.path.join(RES, "F06_strata_AF.csv"), index=False, encoding="utf-8-sig")

# ---------- D. absolute risk and time to diagnosis ----------
ar = []
for k in ["All", "HbA1c <36", "HbA1c 42-47"]:
    s = m[groups[k](m)]
    for t in [1, 3, 5, 10]:
        a, c = cif(s[s.exp == 1], t=t, var=True), cif(s[s.exp == 0], t=t, var=True)
        ar.append(dict(group=k, year=t, AP=f"{a[0]:.1f} ({a[1]:.1f}–{a[2]:.1f})", controls=f"{c[0]:.1f} ({c[1]:.1f}–{c[2]:.1f})", difference=round(a[0] - c[0], 1)))
AR = pd.DataFrame(ar); P(AR.to_string(index=False)); AR.to_csv(os.path.join(RES, "F06_absolute_risk.csv"), index=False, encoding="utf-8-sig")
ev = m[(m.exp == 1) & (m.E == 1)].copy(); yrs = ev["T"] + 90 / 365.25
P(f"Time from AP to diabetes diagnosis (years): median {yrs.median():.1f} (IQR {yrs.quantile(.25):.1f}–{yrs.quantile(.75):.1f}); "
  f"within 1 year {(yrs<=1).mean()*100:.0f}%, 3 years {(yrs<=3).mean()*100:.0f}%, 5 years {(yrs<=5).mean()*100:.0f}%")

# ---------- E. appendicitis and cholecystitis versus their own HbA1c-matched populations (effect of hospitalization itself) ----------
hz = []
for evt, lab in [("appx", "Acute appendicitis"), ("chole", "Acute cholecystitis (no AP)")]:
    ce = exposed(base, evt); mm = follow(match(base, ce, seed=11)); r = cox(mm); h = cox(mm[mm.hcat == "<36"])
    hz.append(dict(exposure=lab, cases=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), HR_lt36=fmt(h), CIF5=f"{cif(mm[mm.exp==1]):.1f} vs {cif(mm[mm.exp==0]):.1f}"))
    P(hz[-1])
pd.DataFrame(hz).to_csv(os.path.join(RES, "F06_comparator_vs_population.csv"), index=False, encoding="utf-8-sig")

# ---------- G. proportional-hazards test ----------
Y = m[["T", "E", "exp", "mid"]].copy(); Y["E"] = (Y.E == 1).astype(int)
fp = CoxPHFitter().fit(Y, "T", "E", strata=["mid"]); ph = proportional_hazard_test(fp, Y, time_transform="rank")
P(f"Proportional-hazards test (Schoenfeld, rank time): P = {ph.summary['p'].iloc[0]:.3f}")

# ---------- H. E-values (HRs and lower CI limits from 02_main.py) ----------
ev_ = lambda hr: hr + np.sqrt(hr * (hr - 1)) if hr > 1 else np.nan
for lab, hr, lo in [("All, matched", 2.74, 2.27), ("All, adjusted", 2.59, 2.01), ("HbA1c <36", 5.08, 3.56), ("vs appendicitis", 2.78, 2.07), ("vs cholecystitis", 1.90, 1.50)]:
    P(f"E-value {lab}: point estimate {ev_(hr):.2f}, lower CI limit {ev_(lo):.2f}")

# ---------- I. covariate missingness ----------
mm_ = covars(m)
P("Covariate missingness (%):", {c: round(mm_[c].isna().mean() * 100, 1) for c in ["hba1c", "bmi", "smoke", "alcohol", "tdi", "ethnic", "tg"]})

# ---------- J. pre-attack HbA1c: AP patients vs participants of the same age and sex (matched on sex + birth year only) ----------
rng = np.random.default_rng(20261011); keep = ["eid", "p31", "p34", "d0", "DM", "dm", "death", "lost", "AP_any", "pcan", "h_cp", "hba1c", "bmi"]
grp = {k: v for k, v in base[keep].groupby(["p31", "p34"])}; ctl = []
for _, c in cases.iterrows():
    gg = grp.get((c.p31, c.p34)); t = c.t0
    ok = gg[(gg.eid != c.eid) & (gg.d0 < c.t_evt) & ~(gg.DM.notna() & (gg.DM <= t)) & ~(gg.dm.notna() & (gg.dm <= t)) & ~(gg.death.notna() & (gg.death <= t))
            & ~(gg.lost.notna() & (gg.lost <= t)) & ~(gg.AP_any.notna() & (gg.AP_any <= t))]
    ctl.append(ok.iloc[rng.choice(len(ok), min(5, len(ok)), replace=False)])
ctl = pd.concat(ctl)
smd = lambda a, b: (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
from scipy.stats import mannwhitneyu
P(f"Pre-attack HbA1c: AP {cases.hba1c.mean():.1f}±{cases.hba1c.std():.1f} vs same age and sex {ctl.hba1c.mean():.1f}±{ctl.hba1c.std():.1f}; SMD {smd(cases.hba1c, ctl.hba1c):.3f}; "
  f"Mann-Whitney P = {mannwhitneyu(cases.hba1c, ctl.hba1c).pvalue:.3f}; ≥42 {(cases.hba1c>=42).mean()*100:.1f}% vs {(ctl.hba1c>=42).mean()*100:.1f}%; ≥39 {(cases.hba1c>=39).mean()*100:.1f}% vs {(ctl.hba1c>=39).mean()*100:.1f}%")
P(f"BMI: AP {cases.bmi.mean():.1f} vs same age and sex {ctl.bmi.mean():.1f}; SMD {smd(cases.bmi, ctl.bmi):.3f}")
out.close()
