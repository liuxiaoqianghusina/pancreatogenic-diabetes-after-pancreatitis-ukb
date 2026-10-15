"""08: Extension analyses.
1a. Full-cohort Cox model with AP as a time-varying exposure (no matching; age as the time scale; exposure starts 90 days after AP,
    and follow-up between AP and AP + 90 days is excluded).
1b. Multiple imputation (MICE, 5 imputations) for the fully adjusted matched model.
2.  Pancreatic intervention as a harder severity marker (necrosectomy J57.6 / drainage J61, J66, J42.4, J60.1, from 3 days before to
    90 days after AP) → dose-response; sensitivity analysis excluding matched sets of patients with pancreatic resection.

Requires cache/matched_main.pkl (02_main.py), cache/extra_codes.pkl (01b_extra_codes.py) and cache/opcs_pancreas.pkl (01c_opcs.py)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from lifelines import CoxPHFitter
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

YR = 365.25

# ======================= 1a. full cohort, time-varying exposure =======================
df = load(); base = base_cohort(df)
b = covars(base.assign(t0=base.d0)); b["hba1c5"] = b.hba1c / 5; b["bmi5"] = b.bmi / 5
stop_ = b[["DM", "death", "lost"]].min(axis=1).fillna(END).clip(upper=END)
cens2 = b[["pcan", "h_cp"]].min(axis=1)                          # censor at chronic pancreatitis / pancreatic cancer
stop_ = stop_.where(~(cens2.notna() & (cens2 < stop_)), cens2)
age0 = (b.d0 - b.birth).dt.days / YR
ap = b.AP.where(b.AP.notna() & (b.AP > b.d0) & (b.AP < stop_))
rows = []
# unexposed segment: baseline → min(AP, stop)
e1 = ap.fillna(stop_); end1 = pd.concat([e1, stop_], axis=1).min(axis=1)
ev1 = (b.DM.notna() & (b.DM <= end1) & (b.DM == stop_) & ap.isna()).astype(int)   # events in the unexposed segment only for people who never had AP
ev1 = ev1 | (ap.notna() & b.DM.notna() & (b.DM <= ap)).astype(int)
seg1 = pd.DataFrame({"eid": b.eid, "start": age0, "stop": age0 + (end1 - b.d0).dt.days / YR, "exp": 0, "E": ev1.astype(int)})
# exposed segment: AP + 90 days → stop (people with diabetes or censoring within 90 days after AP do not enter it)
t90 = ap + pd.Timedelta(days=90); ok = ap.notna() & (t90 < stop_) & ~(b.DM.notna() & (b.DM <= t90))
seg2 = pd.DataFrame({"eid": b.eid[ok], "start": (age0 + (t90 - b.d0).dt.days / YR)[ok], "stop": (age0 + (stop_ - b.d0).dt.days / YR)[ok],
                     "exp": 1, "E": (b.DM.notna() & (b.DM <= stop_))[ok].astype(int)})
tv = pd.concat([seg1, seg2]); tv = tv[tv.stop > tv.start]
COVS = ["male", "hba1c5", "bmi5", "smk_cur", "smk_prev", "alc_cur", "tdi", "white", "ln_tg"]
tv = tv.merge(b[["eid"] + COVS], on="eid", how="left")
print(f"1a. Full cohort: {b.shape[0]} participants; {len(tv)} segments; {int((tv.exp == 1).sum())} exposed segments with {int(tv[tv.exp == 1].E.sum())} events")


def tvcox(d, covs, extra=()):
    x = d[["start", "stop", "E", "eid", "exp"] + list(covs) + list(extra)].dropna()
    f = CoxPHFitter().fit(x.drop(columns=["eid"]), duration_col="stop", event_col="E", entry_col="start", robust=False)
    return f


res = []
for lab, covs in [("Age (time scale) + sex", ["male"]), ("+ HbA1c, BMI", ["male", "hba1c5", "bmi5"]), ("Fully adjusted", COVS)]:
    f = tvcox(tv, covs); s = f.summary.loc["exp"]
    res.append(dict(model=lab, HR=f"{s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}–{s['exp(coef) upper 95%']:.2f})", P=s["p"]))
for lab, lo, hi in [("HbA1c <36", 0, 36), ("HbA1c 36–38", 36, 39), ("HbA1c 39–41", 39, 42), ("HbA1c 42–47", 42, 48)]:
    sub = tv[(tv.hba1c5 * 5 >= lo) & (tv.hba1c5 * 5 < hi)]
    f = tvcox(sub, COVS); s = f.summary.loc["exp"]
    res.append(dict(model=f"Fully adjusted, {lab}", HR=f"{s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}–{s['exp(coef) upper 95%']:.2f})", P=s["p"],
                    exposed_events=int(sub[sub.exp == 1].E.sum())))
tvi = tv.assign(x_h=tv.exp * (tv.hba1c5 - 7)); f = tvcox(tvi, COVS, ["x_h"])
res.append(dict(model="Interaction: AP × HbA1c (per 5 mmol/mol)", HR=f"ratio {np.exp(f.params_['x_h']):.2f}", P=f.summary.loc['x_h', 'p']))
R1 = pd.DataFrame(res); print(R1.to_string(index=False))

# ======================= 1b. multiple imputation (fully adjusted matched model) =======================
m = pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl")); m = covars(m)
ADJ_ = ["hba1c", "bmi", "smk_cur", "smk_prev", "alc_cur", "tdi", "white", "ln_tg"]


def rubin(betas, ses):
    betas, ses = np.array(betas), np.array(ses); qb = betas.mean(); w = (ses ** 2).mean(); bvar = betas.var(ddof=1)
    t = w + (1 + 1 / len(betas)) * bvar; return qb, np.sqrt(t)


def mi_cox(d, x, covs, strata, imp_cols, k=5):
    bs, ss = [], []
    for i in range(k):
        dd = d.copy()
        aux = [c for c in ["hba1c", "bmi", "exp", "male"] if c in dd and c not in imp_cols]
        mat = dd[imp_cols + aux + ["E", "T"]].assign(E=(dd.E == 1).astype(float)).astype(float).values
        imp = IterativeImputer(sample_posterior=True, random_state=i, max_iter=10).fit_transform(mat)
        dd[imp_cols] = imp[:, :len(imp_cols)]
        for c in ["smk_cur", "smk_prev", "alc_cur", "white"]:
            if c in imp_cols: dd[c] = (dd[c] > 0.5).astype(float)
        r = dd[["T", "E", x] + covs + ([strata] if strata else [])].copy(); r["E"] = (r.E == 1).astype(int)
        f = CoxPHFitter(penalizer=0.0).fit(r, "T", "E", strata=[strata] if strata else None)
        bs.append(f.params_[x]); ss.append(f.standard_errors_[x])
    q, se = rubin(bs, ss); return np.exp(q), np.exp(q - 1.96 * se), np.exp(q + 1.96 * se)


hr = mi_cox(m, "exp", ADJ_, "mid", ["smk_cur", "smk_prev", "alc_cur", "tdi", "white", "ln_tg"])
R1b = pd.DataFrame([dict(analysis="Fully adjusted matched model (MICE, 5 imputations)", HR=f"{hr[0]:.2f} ({hr[1]:.2f}–{hr[2]:.2f})",
                         comparison="complete cases 2.59 (2.01–3.33)")])
print(R1b.to_string(index=False))

# ======================= 2. pancreatic intervention as a harder severity marker =======================
X_ = pd.read_pickle(os.path.join(ROOT, "cache", "extra_codes.pkl"))
op = pd.read_pickle(os.path.join(ROOT, "cache", "opcs_pancreas.pkl"))
cs_ = m[m.exp == 1][["mid", "eid", "t_evt", "t0", "etiol"]].merge(op, on="eid", how="left").merge(X_, on="eid", how="left")
win = lambda c, a, bb: cs_[c].notna() & (cs_[c] >= cs_.t_evt + pd.Timedelta(days=a)) & (cs_[c] <= cs_.t_evt + pd.Timedelta(days=bb))
cs_["interv"] = win("op_necro", -3, 90) | win("op_drain", -3, 90)
cs_["resect"] = cs_.op_resect.notna() & (cs_.op_resect <= cs_.t0)
cs_["of_ps"] = win("h_aki", -3, 30) | win("h_resp", -3, 30) | win("h_shock", -3, 30) | win("h_sirs", -3, 30) | win("h_pseudo", -3, 90)
SEV = ["None of these markers", "Organ failure or pseudocyst (no intervention)", "Pancreatic intervention (necrosectomy/drainage)"]
cs_["sev3"] = np.select([cs_.interv, cs_.of_ps], [SEV[2], SEV[1]], SEV[0])
print("2. Severity among AP patients:", cs_.sev3.value_counts().to_dict(), "; pancreatic resection before or during AP:", int(cs_.resect.sum()))
m2 = m.copy(); m2["sev3"] = m2.mid.map(cs_.set_index("mid").sev3); m2["resect_set"] = m2.mid.map(cs_.set_index("mid").resect)
rows = []
for lab in SEV:
    s = m2[m2.sev3 == lab]
    if (s.exp == 1).sum() < 5: continue
    r = cox(s); a = s[s.exp == 1]
    rows.append(dict(group=lab, AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), AP_5y=f"{cif(a):.1f}%"))
# trend within AP (ordinal 0/1/2, adjusted for age, sex, HbA1c, BMI)
a = covars(m2[m2.exp == 1].copy()); a["ord"] = a.sev3.map({lab: i for i, lab in enumerate(SEV)})
a["age_evt"] = (a.t_evt - a.birth).dt.days / YR
rt = cox(a.rename(columns={"ord": "x"}), x="x", covs=["age_evt", "male", "hba1c", "bmi"], strata=None)
rows.append(dict(group="Trend within AP (per level)", AP=f"{rt['n1']}", controls="", HR=fmt(rt), AP_5y=f"P={rt['p']:.1e}"))
r = cox(m2[m2.resect_set != True]); rows.append(dict(group="Main analysis excluding matched sets with pancreatic resection", AP=f"{r['n1']}/{r['e1']}", controls=f"{r['n0']}/{r['e0']}", HR=fmt(r), AP_5y=""))
R2 = pd.DataFrame(rows); print(R2.to_string(index=False))

R1.to_csv(os.path.join(RES, "G07_1a_timevarying.csv"), index=False, encoding="utf-8-sig")
R1b.to_csv(os.path.join(RES, "G07_1b_mice.csv"), index=False, encoding="utf-8-sig")
R2.to_csv(os.path.join(RES, "G07_2_severity_interventions.csv"), index=False, encoding="utf-8-sig")
print("done")
