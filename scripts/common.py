"""Shared functions: cohort construction, matching on pre-attack HbA1c, follow-up, Cox models and cumulative incidence.

Output directories are set with the environment variable PPDM_OUTPUT_DIR (default: the repository root);
cache/, results/ and figures/ are created inside it."""
import os, warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter, AalenJohansenFitter
from lifelines.utils import ConvergenceWarning
warnings.filterwarnings("ignore")

ROOT = os.environ.get("PPDM_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
for _d in ["cache", "figures"]: os.makedirs(os.path.join(ROOT, _d), exist_ok=True)
END = pd.Timestamp("2022-12-31")
HCUT = [0, 36, 39, 42, 48]; HLAB = ["<36", "36-38", "39-41", "42-47"]
BCUT = [0, 25, 30, 200]; BLAB = ["<25", "25-29.9", "≥30"]


def load():
    df = pd.read_pickle(os.path.join(ROOT, "cache", "cohort_full.pkl"))
    df["male"] = (df.p31 == 1).astype(int)
    df["hcat"] = pd.cut(df.hba1c, HCUT, right=False, labels=HLAB)
    df["bcat"] = pd.cut(df.bmi, BCUT, right=False, labels=BLAB)
    df["white"] = (df.ethnic == df.ethnic.mode()[0]).astype(float).where(df.ethnic.notna())
    return df


def base_cohort(df, ap_def="fo", dm_def="fo"):
    """Baseline cohort: no diabetes, HbA1c <48 mmol/mol, HbA1c and BMI available, no AP, chronic pancreatitis or pancreatic cancer before baseline."""
    d = df.copy()
    d["AP"] = d.ap if ap_def == "fo" else d.h_ap
    d["DM"] = d.dm if dm_def == "fo" else d.h_dm
    any_ap = d[["ap", "h_ap"]].min(axis=1)              # AP from any source, used for exclusion and for censoring controls
    d["AP_any"] = any_ap
    ok = (~d.dm_bad) & ~(d.dm.notna() & (d.dm <= d.d0)) & ~(d.h_dm.notna() & (d.h_dm <= d.d0))
    ok &= d.hba1c.notna() & (d.hba1c < 48) & d.bmi.notna() & ~d.ap_bad
    ok &= ~(any_ap.notna() & (any_ap <= d.d0)) & ~(d.h_cp.notna() & (d.h_cp <= d.d0)) & ~(d.pcan.notna() & (d.pcan <= d.d0))
    return d[ok].copy()


def exposed(base, evt, blank=90, excl_ap_window=365, max_gap=None):
    """First event after baseline; index date = event + `blank` days; no diabetes, death, loss to follow-up, pancreatic cancer or chronic pancreatitis by the index date."""
    col = "AP" if evt == "AP" else evt
    e = base[base[col].notna() & (base[col] > base.d0) & (base[col] <= END)].copy()
    if evt != "AP":
        e = e[~(e.AP_any.notna() & (e.AP_any <= e[col] + pd.Timedelta(days=excl_ap_window)))]
    e["t_evt"] = e[col]; e["t0"] = e[col] + pd.Timedelta(days=blank)
    if max_gap is not None:
        e = e[(e.t_evt - e.d0).dt.days / 365.25 <= max_gap]
    bad = (e.DM.notna() & (e.DM <= e.t0)) | (e.dm.notna() & (e.dm <= e.t0)) | (e.death.notna() & (e.death <= e.t0)) \
        | (e.lost.notna() & (e.lost <= e.t0)) | (e.pcan.notna() & (e.pcan <= e.t0)) | (e.h_cp.notna() & (e.h_cp <= e.t0))
    e = e[~bad & (e.t0 < END)].copy()
    e["age_evt"] = (e.t_evt - e.birth).dt.days / 365.25
    return e


def etiology(e):
    bil = e.k851 | (e.gs.notna() & (e.gs <= e.t_evt + pd.Timedelta(days=30)))
    alc = ~bil & (e.k852 | (e.k70.notna() & (e.k70 <= e.t_evt)) | e.f10)
    return pd.Series(np.select([bil, alc], ["biliary", "alcohol"], "other"), index=e.index)


def match(base, cases, k=5, seed=20261011):
    """Incidence density sampling, 1:k exact matching on sex, birth year, HbA1c category and BMI category. Controls must be alive, under
    follow-up and free of diabetes, AP, pancreatic cancer and chronic pancreatitis on the index date, with baseline before the case's event."""
    rng = np.random.default_rng(seed)
    keep = ["eid", "p31", "p34", "hcat", "bcat", "d0", "DM", "dm", "death", "lost", "AP_any", "pcan", "h_cp"]
    grp = {key: g for key, g in base[keep].groupby(["p31", "p34", "hcat", "bcat"], observed=True)}
    recs = []
    for mid, (_, c) in enumerate(cases.iterrows()):
        g = grp.get((c.p31, c.p34, c.hcat, c.bcat))
        if g is None: continue
        t = c.t0
        ok = g[(g.eid != c.eid) & (g.d0 < c.t_evt) & ~(g.DM.notna() & (g.DM <= t)) & ~(g.dm.notna() & (g.dm <= t))
               & ~(g.death.notna() & (g.death <= t)) & ~(g.lost.notna() & (g.lost <= t)) & ~(g.AP_any.notna() & (g.AP_any <= t))
               & ~(g.pcan.notna() & (g.pcan <= t)) & ~(g.h_cp.notna() & (g.h_cp <= t))]
        if len(ok) == 0: continue
        sel = ok.eid.values[rng.choice(len(ok), min(k, len(ok)), replace=False)]
        recs.append((mid, 1, c.eid, t, c.eid))
        recs += [(mid, 0, s, t, c.eid) for s in sel]
    m = pd.DataFrame(recs, columns=["mid", "exp", "eid", "t0", "case_eid"])
    cols = [c for c in base.columns if c not in ("t0",)]
    m = m.merge(base[cols], on="eid", how="left")
    ce = cases.set_index("eid")
    m["t_evt"] = m.case_eid.map(ce.t_evt)
    for v in ["etiol", "age_evt"]:
        if v in ce: m[v] = m.case_eid.map(ce[v])
    return m


def follow(m, outcome="DM", censor_ap_ctrl=True):
    """Follow up to the outcome, death, loss to follow-up or END; controls are censored at a later AP. Returns T (years) and E (1 outcome, 2 death, 0 censored)."""
    m = m.copy()
    stop = m[[outcome, "death", "lost"]].min(axis=1).fillna(END).clip(upper=END)
    if censor_ap_ctrl:
        c = (m.exp == 0) & m.AP_any.notna() & (m.AP_any > m.t0) & (m.AP_any < stop)
        stop = stop.where(~c, m.AP_any)
    ev = np.where(m[outcome].notna() & (m[outcome] <= stop), 1, np.where(m.death.notna() & (m.death <= stop), 2, 0))
    m["E"] = ev; m["T"] = (stop - m.t0).dt.days / 365.25
    return m[m["T"] > 0].copy()


def cox(d, x="exp", covs=(), strata="mid", robust=False):
    dd = d[["T", "E", x] + list(covs) + ([strata] if strata else [])].dropna().copy()
    dd["E"] = (dd.E == 1).astype(int)
    for c in covs:
        if dd[c].std() == 0: dd = dd.drop(columns=c)
    f = CoxPHFitter(penalizer=0.0)
    try:
        f.fit(dd, "T", "E", strata=[strata] if strata else None, robust=robust)
    except Exception:
        f = CoxPHFitter(penalizer=0.0); f.fit(dd, "T", "E", strata=[strata] if strata else None, robust=robust, fit_options={"step_size": 0.2})
    s = f.summary.loc[x]
    return dict(n1=int((dd[x] == 1).sum()), e1=int(dd[dd[x] == 1].E.sum()), n0=int((dd[x] == 0).sum()), e0=int(dd[dd[x] == 0].E.sum()),
                HR=s["exp(coef)"], lo=s["exp(coef) lower 95%"], hi=s["exp(coef) upper 95%"], p=s["p"])


def rate(d):
    return d[d.E == 1].shape[0] / d["T"].sum() * 1000


def cif(d, t=5, var=False):
    d = d.reset_index(drop=True)
    aj = AalenJohansenFitter(calculate_variance=var, seed=1).fit(d["T"], pd.Series(d.E.values), event_of_interest=1)
    s = aj.cumulative_density_; i = s.index[s.index <= t][-1]
    if not var: return float(s.loc[i].iloc[0]) * 100
    c = aj.confidence_interval_cumulative_density_
    return float(s.loc[i].iloc[0]) * 100, float(c.loc[i].iloc[0]) * 100, float(c.loc[i].iloc[1]) * 100


def fmt(r, k=2):
    return f"{r['HR']:.{k}f} ({r['lo']:.{k}f}–{r['hi']:.{k}f})"


def covars(m):
    m = m.copy()
    m["smk_cur"] = (m.smoke == 2).astype(float).where(m.smoke.notna()); m["smk_prev"] = (m.smoke == 1).astype(float).where(m.smoke.notna())
    m["alc_cur"] = (m.alcohol == 2).astype(float).where(m.alcohol.notna())
    m["ln_tg"] = np.log(m.tg)
    m["gs_prior"] = (m.gs.notna() & (m.gs <= m.t0)).astype(int)
    return m

ADJ = ["hba1c", "bmi", "smk_cur", "smk_prev", "alc_cur", "tdi", "white"]
ADJ_FULL = ADJ + ["ln_tg"]
