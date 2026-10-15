# Diabetes After Acute Pancreatitis Is Largely Pancreatogenic: A UK Biobank Cohort Study Using Pre-attack HbA1c, Polygenic Risk, and Active Comparators

Analysis code for the study. Patients with incident acute pancreatitis (AP) in UK Biobank were matched 1:5 to population controls on sex, birth year, pre-attack HbA1c category and BMI category, using incidence-density sampling. The code estimates the risk of incident diabetes from 90 days after the attack, and triangulates the result with:

- a type 2 diabetes polygenic risk score;
- active comparators (acute appendicitis, acute cholecystitis);
- negative-control outcomes;
- a set of sensitivity analyses.

## Data

The repository contains **code only**. No UK Biobank data, individual-level results or manuscript files are included. Access to UK Biobank data requires an approved application (https://www.ukbiobank.ac.uk). The scripts expect four table extracts from the UK Biobank Research Analysis Platform, plus a polygenic score file:

| File in `UKB_DATA_DIR` | Fields used |
|---|---|
| `UKB619657_health_outcomes_2025Nov.csv` | <ul><li>First-occurrence dates p130706–p130714, p131164, p131604, p131658, p131674, p131676, p131682–p131684, p131872</li><li>Hospital inpatient codes p41270 / p41271 / p41272, with their first-date arrays p41280 / p41281 / p41282</li><li>Cancer registry p40005 / p40006</li><li>Death date p40000</li><li>Primary care records p42040</li></ul> |
| `UKB619657_population_characteristics_2025Nov.csv` | p31 sex, p34 / p52 birth year and month, p191 date lost to follow-up, p22189 Townsend index |
| `UKB619657_biological_samples_2025Nov.csv` | p30750 HbA1c, p30740 glucose, p30870 triglycerides |
| `UKB619657_assessment_center_2025Nov.csv` | p53 assessment date, p21000 ethnicity, p21001 BMI, p20116 smoking status, p20117 alcohol drinker status |

The polygenic score file (`UKB_PRS_FILE`) needs these columns:

- `eid`
- `PRScs_z`: a type 2 diabetes PRScs score, standardized
- `Batch`
- `pc1`–`pc10`: genetic principal components

## Requirements

- Python 3.9 or later, with the packages in `requirements.txt`. The analyses were run with lifelines 0.30.3, pandas 2.3.3, numpy 2.4.6, scikit-learn 1.9.0, scipy 1.17.1 and matplotlib 3.11.0.
- R 4.2 or later with the `survival` package. The analyses were run with R 4.2.2 and survival 3.4.0.
- The figures use the Arial font.

## Usage

```bash
export UKB_DATA_DIR=/path/to/ukb_tables
export UKB_PRS_FILE=/path/to/t2d_prs.csv
export PPDM_OUTPUT_DIR=/path/to/output     # cache/, results/, figures/, tables/ and supplement/ are created here
bash run_all.sh
```

The full pipeline takes about 1–2 hours on a laptop. Most of the time goes to bootstrap resampling (300 replicates by default; pass a smaller number to `02_main.py`, `06_additional.py` or `11_prs.py` for a quick run).

Controls are drawn at random within matching cells, so the selected controls depend on the row order of the cohort table. Rows are sorted by `eid`. To reproduce a given set of controls exactly, set `PPDM_ROW_ORDER` to a one-column CSV of `eid`s that gives the row order; leave it unset otherwise.

## Scripts

Scripts are numbered in run order. Each later script reads the cached cohort and the matched cohort written by the earlier ones.

| Script | Content |
|---|---|
| `common.py` | Shared functions: baseline cohort, AP cases, matching on pre-attack HbA1c and BMI, follow-up, Cox models, Aalen-Johansen cumulative incidence |
| `01_load.py` | Builds the analysis cohort from the raw UK Biobank tables |
| `01b_extra_codes.py`, `01c_opcs.py` | Organ failure / pseudocyst codes and pancreatic procedures (severity markers) |
| `02_main.py` | Main matched analysis, HbA1c strata, additive and multiplicative interaction, subgroups, active comparators |
| `03_sensitivity.py`, `04_finegray.R` | Sensitivity analyses, negative-control outcomes, Fine-Gray competing-risk models |
| `05_conventional_matching.py` | Matching on age and sex only, as in previous studies |
| `06_additional.py`, `06b_continuous.py` | Flow chart counts, attributable fractions, absolute risks, E-values, continuous HbA1c interaction |
| `07_severity.py`, `08_extensions.py` | Severity proxies, pancreatic interventions, full-cohort time-varying exposure model, multiple imputation |
| `09_early_window.py`, `10_gp_linked.py` | Early post-attack window; participants with linked primary care records |
| `11_prs.py` | Type 2 diabetes polygenic risk: validity, additional matching on PRS tertile, interaction, joint strata |
| `12_table1.py`, `13_figures.py`, `14_graphical_abstract.py`, `15_supplement.py` | Table 1, Figures 1–3, graphical abstract, supplemental figure and tables |
| `md2docx.py` | Markdown-to-Word converter used for the supplement |

## License

MIT; see `LICENSE`.
