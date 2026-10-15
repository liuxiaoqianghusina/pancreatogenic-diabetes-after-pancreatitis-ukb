"""12: Table 1 (characteristics of the main matched cohort), computed from cache/matched_main.pkl and written to tables/table1.md."""
import numpy as np, pandas as pd
from common import *
m = covars(pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl")))
prs = pd.read_csv(os.environ.get("UKB_PRS_FILE", "../ukb_data/T2D_PRScs_zscore.csv"), usecols=["eid", "PRScs_z"])
m = m.merge(prs, on="eid", how="left")
base = base_cohort(load()).merge(prs, on="eid", how="left"); mu, sd = base.PRScs_z.mean(), base.PRScs_z.std()
m["prs"] = (m.PRScs_z - mu) / sd
def n_pct(s): return f"{int(s.sum()):,} ({s.mean()*100:.1f})"
def mean_sd(s, k=1): return f"{s.mean():.{k}f} ± {s.std():.{k}f}"
def med_iqr(s, k=1): return f"{s.median():.{k}f} ({s.quantile(.25):.{k}f}–{s.quantile(.75):.{k}f})"
pct = lambda v: v / 10.929 + 2.15   # NGSP: mmol/mol → %
def hb_dual(s): return f"{pct(s.median()):.1f} ({pct(s.quantile(.25)):.1f}–{pct(s.quantile(.75)):.1f}) [{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})]"
rows = []
for lab, f in [("No.", lambda d: f"{len(d):,}"), ("Age at index date, years", lambda d: mean_sd(d.age_t0)), ("Women", lambda d: n_pct(d.male == 0)),
               ("White ethnicity", lambda d: n_pct(d.white == 1)), ("Townsend deprivation index", lambda d: mean_sd(d.tdi, 2)),
               ("Current smoking", lambda d: n_pct(d.smk_cur == 1)), ("Current alcohol drinking", lambda d: n_pct(d.alc_cur == 1)),
               ("BMI, kg/m²", lambda d: mean_sd(d.bmi)), ("  <25", lambda d: n_pct(d.bcat == "<25")), ("  25–29.9", lambda d: n_pct(d.bcat == "25-29.9")), ("  ≥30", lambda d: n_pct(d.bcat == "≥30")),
               ("Pre-attack HbA1c, % [mmol/mol]", lambda d: hb_dual(d.hba1c)), ("  <5.4% (<36)", lambda d: n_pct(d.hcat == "<36")), ("  5.4–5.6% (36–38)", lambda d: n_pct(d.hcat == "36-38")),
               ("  5.7–5.9% (39–41)", lambda d: n_pct(d.hcat == "39-41")), ("  6.0–6.4% (42–47)", lambda d: n_pct(d.hcat == "42-47")),
               ("Triglycerides, mmol/L", lambda d: med_iqr(d.tg, 2)), ("Cholelithiasis before index date", lambda d: n_pct(d.gs_prior == 1)),
               ("Type 2 diabetes PRS, z score", lambda d: f"{d.prs.mean():.2f} ± {d.prs.std():.2f} (n={int(d.prs.notna().sum()):,})"),
               ("Follow-up, years", lambda d: med_iqr(d["T"])), ("Incident diabetes", lambda d: n_pct(d.E == 1))]:
    rows.append([lab, f(m[m.exp == 1]), f(m[m.exp == 0])])
T = pd.DataFrame(rows, columns=["Characteristic", "Acute pancreatitis", "Matched controls"])
md = "## Table 1—Characteristics of patients with acute pancreatitis and matched controls at the index date\n\n| " + " | ".join(T.columns) + " |\n|---|---|---|\n" + "".join("| " + " | ".join(r) + " |\n" for r in T.values)
md += ("\nData are mean ± SD, median (IQR), or n (%). The index date was 90 days after acute pancreatitis; controls shared the index date of their matched case "
       "and were matched 1:5 on sex, birth year, pre-attack HbA1c category, and BMI category. HbA1c, BMI, smoking, alcohol drinking, and triglycerides were measured at the "
       "UK Biobank baseline assessment (2006–2010). HbA1c is shown as % with mmol/mol in square brackets. The PRS was standardized in the full cohort and is available for "
       "genotyped participants only. PRS, polygenic risk score.\n")
TAB = os.path.join(ROOT, "tables"); os.makedirs(TAB, exist_ok=True)
open(os.path.join(TAB, "table1.md"), "w").write(md); print(T.to_string(index=False))
