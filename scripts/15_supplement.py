"""15: Online-only supplemental material: Supplemental Fig. S1 (flow chart) and Supplemental Tables S1–S10, generated from the CSV files
in results/ and numbered in order of first citation in the paper.
Writes supplement/Supplemental_Material.md/.docx and figures/DC_FigS1_flow.png. Prints any flow-chart text that overruns its box."""
import re, os, subprocess, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import ROOT, RES
R = lambda f: pd.read_csv(os.path.join(RES, f))
os.makedirs(os.path.join(ROOT, "supplement"), exist_ok=True)
OUT = os.path.join(ROOT, "supplement", "Supplemental_Material.md")
pct = lambda v: v / 10.929 + 2.15
HB = {"<36": "<5.4% (<36)", "36-38": "5.4–5.6% (36–38)", "39-41": "5.7–5.9% (39–41)", "42-47": "6.0–6.4% (42–47)"}
def hb_label(s):
    for k, v in HB.items():
        if k in s: return "HbA1c " + v
    return s
def md(df, cols=None):
    df = df.fillna("").astype(str)
    if cols: df.columns = cols
    df = df.apply(lambda c: c.str.replace(r"(?<![\d.])-(\d)", r"−\1", regex=True))
    return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "".join("| " + " | ".join(r) + " |\n" for r in df.values)
def pfmt(p):
    p = float(p)
    return "<0.001" if p < 0.001 else f"{p:.3f}"
S = ["# Online-Only Supplemental Material", "",
     "Diabetes After Acute Pancreatitis Is Largely Pancreatogenic: A UK Biobank Cohort Study Using Pre-attack HbA1c, Polygenic Risk, and Active Comparators", ""]

# ---------- Supplemental Fig. S1: flow chart ----------
fl = R("F06_flow.csv")
EN = ["UK Biobank participants", "Diabetes at baseline (incl. undated)", "Missing HbA1c", "HbA1c ≥6.5% (≥48 mmol/mol)",
      "Missing BMI", "AP before baseline (incl. undated)", "Chronic pancreatitis before baseline", "Pancreatic cancer before baseline",
      "First AP after baseline (to 31 Dec 2022)", "Diabetes before or ≤90 days after AP", "Death/loss to follow-up ≤90 days",
      "Pancreatic cancer/CP by index date", "Index date after end of follow-up"]
n = [int(str(v).replace(",", "")) for v in fl["n"].iloc[:13]]
plt.rcParams.update({"font.family": "Arial", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.0, 7.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off"); boxes = []
def box(x, y, w, h, t, bold=False, left=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec="#333333", lw=0.8))
    tx = ax.text(x + 0.18 if left else x + w / 2, y + h / 2, t, ha="left" if left else "center", va="center", fontsize=8,
                 fontweight="bold" if bold else "normal", linespacing=1.35); boxes.append((tx, x, w))
def arrow(x0, y0, x1, y1): ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))
box(0.3, 11.8, 4.4, 0.9, f"{EN[0]}\nn = {n[0]:,}", True)
box(5.1, 8.85, 4.75, 2.75, "Excluded:\n" + "\n".join(f"{EN[i]}, n = {n[i-1]-n[i]:,}" for i in range(1, 8)), left=True)
arrow(2.5, 11.8, 2.5, 9.6); ax.plot([2.5, 5.1], [10.22, 10.22], color="#333333", lw=0.8)
box(0.3, 8.7, 4.4, 0.9, f"Eligible participants\nn = {n[7]:,}", True)
arrow(2.5, 8.7, 2.5, 7.6)
box(0.3, 6.7, 4.4, 0.9, f"{EN[8]}\nn = {n[8]:,}", True)
box(5.1, 4.5, 4.75, 1.95, "Excluded:\n" + "\n".join(f"{EN[i]}, n = {n[i-1]-n[i]:,}" for i in range(9, 13)), left=True)
arrow(2.5, 6.7, 2.5, 4.1); ax.plot([2.5, 5.1], [5.47, 5.47], color="#333333", lw=0.8)
box(0.3, 3.2, 4.4, 0.9, f"Patients with AP\nn = {n[12]:,}", True)
arrow(2.5, 3.2, 2.5, 2.1)
box(0.3, 0.75, 9.55, 1.35, "Matched cohort: 2,072 patients with AP and 10,360 controls\n(1:5 on sex, birth year, pre-attack HbA1c category, and BMI category)\n"
    "Polygenic risk analysis: 1,583 patients and 7,906 controls\nadditionally matched on PRS tertile", True)
fig.tight_layout(); fig.canvas.draw(); rr = fig.canvas.get_renderer()
over = [t.get_text()[:30] for t, x, w in boxes if t.get_window_extent(rr).transformed(ax.transData.inverted()).x1 > x + w - 0.05]
print("Flow-chart text overrunning a box:", over if over else "none")
fsv = os.path.join(ROOT, "figures", "DC_FigS1_flow.png"); fig.savefig(fsv, dpi=600); fig.savefig(fsv.replace(".png", ".pdf")); plt.close(fig)
S += ["## Supplemental Figures", "", f"![Supplemental Fig. S1]({fsv})", "",
      "Supplemental Figure S1—Flow of participants. AP, acute pancreatitis; CP, chronic pancreatitis; PRS, polygenic risk score.", "", "\\pagebreak", "", "# Tables", ""]

# ---------- Table S1: code definitions ----------
codes = [["Acute pancreatitis", "ICD-10 K85; ICD-9 577.0", "First-occurrence field 131682 (hospital, primary care, self-report, death)"],
         ["Biliary etiology", "K85.1, or cholelithiasis (K80) before or ≤30 days after AP", "Hospital records; field 131674"],
         ["Alcohol-related etiology", "K85.2, or alcoholic liver disease (K70) before AP, or F10 in hospital records", "Hospital records; field 131658"],
         ["Chronic pancreatitis", "ICD-10 K86.0, K86.1; ICD-9 577.1", "Hospital records"],
         ["Pancreatic cancer", "ICD-10 C25", "Cancer registry and hospital records"],
         ["Diabetes (outcome and exclusion)", "ICD-10 E10–E14; ICD-9 250", "First-occurrence fields 130706–130714 and hospital records"],
         ["Acute appendicitis", "ICD-10 K35", "First-occurrence field 131604"],
         ["Acute cholecystitis", "ICD-10 K81, with no AP before or within 1 year after", "First-occurrence field 131676"],
         ["Organ failure", "N17, J96, R57, R65 from 3 days before to 30 days after AP", "Hospital records (first recorded date)"],
         ["Pancreatic pseudocyst", "K86.3 from 3 days before to 90 days after AP", "Hospital records"],
         ["Pancreatic intervention", "OPCS-4 J57.6 (necrosectomy); J61, J66, J42.4, J60.1 (drainage), from 3 days before to 90 days after AP", "Hospital operation records"],
         ["Pancreatic resection", "OPCS-4 J55, J56, J57 (excluding J57.6), J58", "Hospital operation records"],
         ["Cholecystectomy", "OPCS-4 J18", "Hospital operation records"],
         ["Senile cataract (negative control)", "ICD-10 H25", "First-occurrence field 131164"],
         ["Knee osteoarthritis (negative control)", "ICD-10 M17", "First-occurrence field 131872"],
         ["HbA1c, triglycerides", "Baseline biochemistry", "Fields 30750, 30870"]]
S += ["Supplemental Table S1—Definitions of diseases, procedures, and measurements", "", md(pd.DataFrame(codes), ["Variable", "Codes or definition", "Source in UK Biobank"]),
      "AP, acute pancreatitis.", "", "\\pagebreak", ""]

# ---------- Table S2: absolute risk ----------
a = R("F06_absolute_risk.csv"); a["group"] = a["group"].map(hb_label)
S += ["Supplemental Table S2—Cumulative incidence of diabetes (%) at 1, 3, 5, and 10 years after the index date", "",
      md(a, ["Stratum", "Years", "Acute pancreatitis, % (95% CI)", "Matched controls, % (95% CI)", "Difference, percentage points"]),
      "HbA1c is shown as % (mmol/mol). Cumulative incidence was estimated with the Aalen-Johansen estimator, with death as a competing event.", "", "\\pagebreak", ""]

# ---------- Table S3: matching strategies used in previous studies ----------
c = R("F05_conventional_matching.csv")
c["matching"] = ["Sex and birth year (as in previous studies)", "+ BMI category", "+ HbA1c and BMI categories (this study)"]
for col in ["AP_HbA1c_median", "ctrl_HbA1c_median"]: c[col] = c[col].map(lambda v: f"{pct(v):.1f} ({v:.1f})")
c["incidence"] = c["incidence"].str.replace(" vs ", " vs. "); c["CIF5"] = c["CIF5"].str.replace(" vs ", " vs. ")
S += ["Supplemental Table S3—Effect of the matching strategy on the estimated association", "",
      md(c, ["Matching factors", "Median HbA1c, AP, % (mmol/mol)", "Median HbA1c, controls, % (mmol/mol)", "Mean BMI, AP", "Mean BMI, controls",
             "HbA1c ≥6.0%, AP", "HbA1c ≥6.0%, controls", "Incidence per 1,000 PY, AP vs. controls", "HR (95% CI)", "5-year incidence, AP vs. controls"]),
      "All analyses include the same 2,072 patients with AP; only the matching of controls differs. AP, acute pancreatitis; PY, person-years.", "", "\\pagebreak", ""]

# ---------- Table S4: subgroups ----------
sg = R("F02_subgroups.csv")
SG = {"Etiology biliary": "Etiology: biliary", "Etiology alcohol": "Etiology: alcohol related", "Etiology other": "Etiology: other",
      "Follow-up 0–1 year after index": "Follow-up: 0–1 year", "Follow-up 1–3 years": "Follow-up: 1–3 years", "Follow-up >3 years": "Follow-up: >3 years",
      "Men": "Men", "Women": "Women", "Age at AP <65": "Age at AP <65 years", "Age at AP ≥65": "Age at AP ≥65 years",
      "BMI <25": "BMI <25 kg/m²", "BMI 25-29.9": "BMI 25–29.9 kg/m²", "BMI ≥30": "BMI ≥30 kg/m²"}
sg["subgroup"] = sg["subgroup"].map(SG); sg["P"] = sg["P"].map(pfmt)
S += ["Supplemental Table S4—Subgroup analyses", "", md(sg, ["Subgroup", "AP, n/events", "Controls, n/events", "HR (95% CI)", "*P*"]),
      "HRs from Cox models stratified by matched set. AP, acute pancreatitis.", "", "\\pagebreak", ""]

# ---------- Table S5: strata and attributable fractions ----------
f6 = R("F06_strata_AF.csv")
F6 = {"All": "All", "ADA normal <39": "ADA normoglycemia, <5.7% (<39)", "ADA prediabetes 39-47": "ADA prediabetes, 5.7–6.4% (39–47)",
      "NICE normal <42": "NICE normoglycemia, <6.0% (<42)", "NICE prediabetes 42-47": "NICE prediabetes, 6.0–6.4% (42–47)"}
f6["group"] = f6["group"].map(lambda s: F6.get(s, hb_label(s)))
S += ["Supplemental Table S5—HRs, 5-year risk differences, and attributable fractions by pre-attack HbA1c", "",
      md(f6, ["Stratum, % (mmol/mol)", "AP, n/events", "Controls, n/events", "HR (95% CI)", "5-year incidence, AP, %", "5-year incidence, controls, %",
              "Risk difference, percentage points (95% CI)", "Attributable fraction among AP, % (95% CI)", "Number needed to screen over 5 years"]),
      "The attributable fraction is the risk difference divided by the 5-year cumulative incidence in the AP group. CIs from 300 bootstrap samples of matched sets. "
      "Number needed to screen is the number of AP survivors followed for 5 years to detect one case of diabetes. ADA, American Diabetes Association; AP, acute pancreatitis; NICE, National Institute for Health and Care Excellence.",
      "", "\\pagebreak", ""]

# ---------- Table S6: polygenic risk ----------
ps = R("H01_prs_strata.csv"); ps["PRS_tertile"] = ps["PRS_tertile"].map({"All": "All", "T1 low": "Low tertile", "T2 intermediate": "Intermediate tertile", "T3 high": "High tertile"})
ps["incidence"] = ps["incidence"].str.replace(" vs ", " vs. "); ps["P"] = ps["P"].map(pfmt)
jt = R("H01_joint_hba1c_prs.csv"); jt["HbA1c"] = jt["HbA1c"].map({"HbA1c <39": "<5.7% (<39)", "HbA1c 39–47": "5.7–6.4% (39–47)"})
jt["PRS"] = jt["PRS"].map({"T1 low": "Low", "T2 intermediate": "Intermediate", "T3 high": "High"}); jt["CIF5"] = jt["CIF5"].str.replace(" vs ", " vs. ")
sm = open(os.path.join(RES, "H01_summary.txt")).read()
hr_sd = re.search(r"HR (\d\.\d\d) \((\d\.\d\d)–(\d\.\d\d)\)", sm).groups(); hr_sd2 = re.search(r"adjusted for HbA1c and BMI: HR (\d\.\d\d) \((\d\.\d\d)–(\d\.\d\d)\)", sm).groups()
S += ["Supplemental Table S6—Type 2 diabetes polygenic risk score: modification of the association between acute pancreatitis and diabetes", "",
      "A. By PRS tertile (cases additionally matched on PRS tertile)", "",
      md(ps, ["PRS stratum", "AP, n/events", "Controls, n/events", "Incidence per 1,000 PY, AP vs. controls", "HR (95% CI)", "*P*", "5-year incidence, AP, %",
              "5-year incidence, controls, %", "Risk difference, percentage points (95% CI)", "Attributable fraction, % (95% CI)"]),
      "B. Joint stratification by pre-attack HbA1c and PRS tertile", "",
      md(jt, ["Pre-attack HbA1c, % (mmol/mol)", "PRS tertile", "AP, n/events", "Controls, n/events", "HR (95% CI)", "5-year incidence, AP vs. controls, %", "Risk difference, percentage points"]),
      f"The PRS predicted incident diabetes in the full cohort (HR per SD {hr_sd[0]} [{hr_sd[1]}–{hr_sd[2]}]; {hr_sd2[0]} [{hr_sd2[1]}–{hr_sd2[2]}] after adjustment for HbA1c and BMI). "
      "Multiplicative interaction per SD of PRS: ratio of HRs 0.72 (0.58–0.91), *P* = 0.005; RERI (high vs. low tertile) −1.33 (−4.02 to 0.76). "
      "Strata with HbA1c 5.7–6.4% contain few events and are imprecise. AP, acute pancreatitis; PRS, polygenic risk score; PY, person-years; RERI, relative excess risk due to interaction.",
      "", "\\pagebreak", ""]

# ---------- Table S7: severity ----------
sv = R("G07_2_severity_interventions.csv")
sv["group"] = sv["group"].map({"None of these markers": "No organ failure, pseudocyst, or intervention", "Organ failure or pseudocyst (no intervention)": "Organ failure or pseudocyst (no intervention)",
                              "Pancreatic intervention (necrosectomy/drainage)": "Pancreatic necrosectomy or drainage", "Trend within AP (per level)": "Trend within AP (per severity level)",
                              "Main analysis excluding matched sets with pancreatic resection": "Main analysis excluding matched sets with pancreatic resection"})
sv["HR"] = sv["HR"].astype(str); sv["AP_5y"] = sv["AP_5y"].fillna("").astype(str).str.replace(r"^P=(.+)$", lambda x: "*P* < 0.001" if float(x.group(1)) < 0.001 else f"*P* = {float(x.group(1)):.3f}", regex=True)
sv.loc[sv["group"].str.startswith("Trend"), "AP"] = "2,072"   # all matched AP patients
S += ["Supplemental Table S7—Severity of acute pancreatitis and risk of diabetes", "",
      md(sv, ["Group", "AP, n/events", "Controls, n/events", "HR (95% CI)", "5-year incidence, AP / note"]),
      "Severity was classified hierarchically: pancreatic intervention > organ failure or pseudocyst > neither (definitions in Supplemental Table S1). "
      "The trend HR was estimated within patients with AP, adjusted for age, sex, HbA1c, and BMI. The intervention group is small and its estimate imprecise. AP, acute pancreatitis.",
      "", "\\pagebreak", ""]

# ---------- Table S8: active comparators ----------
ac = R("F02_active.csv")
def act_lab(s):
    s = s.replace("biliary AP vs cholecystitis", "Biliary AP vs. acute cholecystitis").replace("vs appendicitis", "AP vs. acute appendicitis")
    s = s.replace("vs cholecystitis", "AP vs. acute cholecystitis").replace("vs both", "AP vs. both comparators")
    for k, v in HB.items(): s = s.replace(f"HbA1c {k}", f"HbA1c {v}")
    return s.replace(" | ", ", ")
ac["comparison"] = ac["comparison"].map(act_lab); ac["P"] = ac["P"].map(pfmt)
pop = R("F06_comparator_vs_population.csv"); pop["exposure"] = ["Acute appendicitis vs. matched population", "Acute cholecystitis vs. matched population"]
pop["HR_lt36"] = pop["HR_lt36"].astype(str); pop["CIF5"] = pop["CIF5"].str.replace(" vs ", " vs. ")
S += ["Supplemental Table S8—Active-comparator analyses", "", "A. Acute pancreatitis vs. acute appendicitis and acute cholecystitis", "",
      md(ac, ["Comparison, model", "AP, n/events", "Comparator, n/events", "HR (95% CI)", "*P*"]),
      "B. Comparator conditions vs. their own HbA1c- and BMI-matched population controls (effect of hospitalization alone)", "",
      md(pop, ["Comparison", "Comparator, n/events", "Controls, n/events", "HR (95% CI)", "HR at HbA1c <5.4% (<36)", "5-year incidence, %"]),
      "Fully adjusted models include age, sex, HbA1c, BMI, smoking, alcohol drinking, Townsend deprivation index, ethnicity, and triglycerides. "
      "Cholecystectomy within 180 days: 67.1% in biliary AP vs. 63.6% in acute cholecystitis. AP, acute pancreatitis.", "", "\\pagebreak", ""]

# ---------- Table S9: negative-control outcomes and E-values ----------
nc = R("F03_negative_control.csv"); nc["outcome"] = ["Senile cataract (H25)", "Knee osteoarthritis (M17)"]
f6s = open(os.path.join(RES, "F06_summary.txt")).read()
ev = re.findall(r"E-value (.+?): point estimate ([\d.]+), lower CI limit ([\d.]+)", f6s)
EVL = {"All, matched": "Main analysis, matched", "All, adjusted": "Main analysis, adjusted", "HbA1c <36": "HbA1c <5.4% (<36)", "vs appendicitis": "AP vs. acute appendicitis", "vs cholecystitis": "AP vs. acute cholecystitis"}
evd = pd.DataFrame([[EVL.get(a, a), b, c] for a, b, c in ev])
S += ["Supplemental Table S9—Negative-control outcomes and E-values", "", "A. Negative-control outcomes", "",
      md(nc, ["Outcome", "AP, n/events", "Controls, n/events", "HR (95% CI)", "*P*"]), "B. E-values", "",
      md(evd, ["Estimate", "E-value, point estimate", "E-value, lower confidence limit"]),
      "Negative-control analyses excluded participants with the outcome before the index date. AP, acute pancreatitis.", "", "\\pagebreak", ""]

# ---------- Table S10: sensitivity analyses ----------
se = R("F03_sensitivity.csv")
SE = {"Main analysis": "Main analysis", "S1 AP and diabetes from hospital records (HES) only": "AP and diabetes from hospital records only", "S2 Diabetes from hospital records only": "Diabetes from hospital records only",
      "S3 AP within 5 years of HbA1c measurement": "AP ≤5 years after HbA1c measurement", "S4 AP within 3 years of HbA1c measurement": "AP ≤3 years after HbA1c measurement",
      "S5 180-day blanking period": "180-day blanking period", "S6 Excluding pancreatic cancer within 2 years of the index date": "Excluding pancreatic cancer ≤2 years after index date",
      "S7 Censoring at chronic pancreatitis": "Censoring at chronic pancreatitis", "S8 1:10 matching": "1:10 matching", "S9 E10 codes not counted as outcome": "Type 1 diabetes codes (E10) not counted"}
se["analysis"] = se["analysis"].map(SE)
se = se.rename(columns={"P_interaction": "Pint"}); se["Pint"] = se["Pint"].map(lambda v: "<0.001" if float(v) < 0.001 else f"{float(v):.3f}")
fg = R("F04_finegray.csv"); tv = R("G07_1a_timevarying.csv"); mi = R("G07_1b_mice.csv"); gp = R("F11_gp_linked.csv"); ew = R("F09_early_window.csv")
extra = pd.DataFrame([["Fine–Gray model (subdistribution HR)", "2,072/179", "10,360/381", fg.SHR.iloc[0], fg.SHR.iloc[1], fg.SHR.iloc[4], ""],
                      ["Full cohort, time-varying exposure, fully adjusted", "", "", tv.HR.iloc[2], tv.HR.iloc[3], tv.HR.iloc[6], "<0.001"],
                      ["Multiple imputation of covariates (adjusted)", "2,072/179", "10,360/381", mi.HR.iloc[0], "", "", ""],
                      ["Primary-care–linked participants", gp.AP.iloc[0], gp["controls"].iloc[0], gp.HR.iloc[0], gp.HR.iloc[1], gp.HR.iloc[4], ""],
                      ["No blanking period, days 0–90", ew.AP.iloc[0], ew["controls"].iloc[0], ew.HR.iloc[0], ew.HR_lt36.iloc[0], "", ""],
                      ["No blanking period, after day 90", ew.AP.iloc[1], ew["controls"].iloc[1], ew.HR.iloc[1], ew.HR_lt36.iloc[1], "", ""]],
                     columns=se.columns)
se = pd.concat([se, extra], ignore_index=True)
S += ["Supplemental Table S10—Sensitivity analyses", "",
      md(se, ["Analysis", "AP, n/events", "Controls, n/events", "HR, all (95% CI)", "HR, HbA1c <5.4% (<36)", "HR, HbA1c 6.0–6.4% (42–47)", "*P* for interaction"]),
      "Unless stated otherwise, HRs are from Cox models stratified by matched set. *P* for interaction tests the product of acute pancreatitis and continuous HbA1c. "
      "The full-cohort model used age as the time scale and treated acute pancreatitis as a time-varying exposure. AP, acute pancreatitis.", ""]
open(OUT, "w", encoding="utf-8").write("\n".join(S))
env = dict(os.environ, DOC_SPACING="1.15")
subprocess.run(["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "md2docx.py"), OUT, OUT.replace(".md", ".docx")], check=True, env=env)
print("supplement ok")
