"""18: summary numbers, Supplementary Tables S11-S12 and Additional file 2 for the NMR analysis.
Reads results/H03_nmr_interaction.csv and H03_nmr_3way.csv from 17_nmr_models.R; measure names come from the UK Biobank RAP data dictionary
(UKB_DATA_DICTIONARY; field codes are used if it is not set); writes results/H03_summary.txt,
results/H03_tableS11.csv, results/H03_tableS12.csv and results/Additional_file_2_NMR.xlsx."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RES

DIC = os.environ.get("UKB_DATA_DICTIONARY")
if DIC and os.path.exists(DIC):
    d = pd.read_csv(DIC, dtype=str)[["name", "title"]]
    title = dict(zip(d.name.str.replace("_i0", "", regex=False), d.title.str.replace(r"\s*\|\s*Instance 0", "", regex=True)))
else:
    title = {}
r = pd.read_csv(os.path.join(RES, "H03_nmr_interaction.csv")); t3 = pd.read_csv(os.path.join(RES, "H03_nmr_3way.csv")).set_index("code")
m = pd.read_csv(os.path.join(ROOT, "cache", "H03_nmr_matched.csv"), usecols=["eid", "exp", "chol_med", "p23407"])
S = {k: v.set_index("code") for k, v in r.groupby("set")}
a = S["all"]
hr = lambda b, s: f"{np.exp(b):.2f} ({np.exp(b - 1.96 * s):.2f}–{np.exp(b + 1.96 * s):.2f})"
pf = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
out = []
P = lambda *x: out.append(" ".join(str(v) for v in x))
P(f"matched rows {len(m):,}; with NMR {int(m.p23407.notna().sum()):,} ({m.p23407.notna().mean()*100:.1f}%); cholesterol-lowering drugs {m.chol_med.mean()*100:.1f}%")
P(f"analysis set (matched sets with a case and >=1 control with NMR): n {int(a.n.iloc[0]):,}; AP {int(a.n_AP.iloc[0]):,} (events {int(a.ev_AP.iloc[0])}); "
  f"controls {int(a.n.iloc[0] - a.n_AP.iloc[0]):,} (events {int(a.ev_ctrl.iloc[0])})")
P(f"measures {len(a)}; controls FDR<0.05 {int((a.FDR_ctrl < .05).sum())}; AP FDR<0.05 {int((a.FDR_AP < .05).sum())}; AP P<0.05 {int((a.P_AP < .05).sum())}")
P(f"median |HR-1| as HR: controls {np.exp(np.median(abs(a.logHR_ctrl))):.2f}; AP {np.exp(np.median(abs(a.logHR_AP))):.2f}")
sig = a.FDR_int < .05
P(f"interaction FDR<0.05 {int(sig.sum())}; P<0.05 {int((a.P_int < .05).sum())}; AP estimate closer to 1 in {int((abs(a.logHR_AP[sig]) < abs(a.logHR_ctrl[sig])).sum())} of {int(sig.sum())}")
b = np.polyfit(a.logHR_ctrl, a.logHR_AP, 1)[0]; P(f"slope of log HR in AP on log HR in controls {b:.2f}; correlation {np.corrcoef(a.logHR_ctrl, a.logHR_AP)[0, 1]:.2f}")
for s in ["biliary", "non-biliary", "adjusted for fasting time", "adjusted for cholesterol-lowering drugs", "no cholesterol-lowering drugs",
          "biliary, no cholesterol-lowering drugs", "inverse normal transformation"]:
    x = S[s]; da = (a.logHR_AP - a.logHR_ctrl)[sig]; dx = (x.logHR_AP - x.logHR_ctrl)[sig]
    P(f"[{s}] n {int(x.n.iloc[0]):,}; AP {int(x.n_AP.iloc[0]):,}; events AP {int(x.ev_AP.iloc[0])}, controls {int(x.ev_ctrl.iloc[0])}; interaction FDR<0.05 {int((x.FDR_int < .05).sum())}, "
      f"P<0.05 {int((x.P_int < .05).sum())}; controls FDR<0.05 {int((x.FDR_ctrl < .05).sum())}; same direction as main among the {int(sig.sum())}: {(np.sign(da) == np.sign(dx)).mean()*100:.0f}%; "
      f"slope vs main {np.polyfit(da, dx, 1)[0]:.2f}; overlap of FDR<0.05 with main {int(((x.FDR_int < .05) & sig).sum())}")
P(f"3-way P<0.05 {int((t3.P_3way < .05).sum())}; FDR<0.05 {int((t3.FDR_3way < .05).sum())}")
off = a[~(np.exp(a.logHR_ctrl).between(0.5, 2.2) & np.exp(a.logHR_AP).between(0.55, 1.8))]
for c, row in off.iterrows(): P(f"off scale in Fig 4A: {title.get(c, c)}, HR controls {np.exp(row.logHR_ctrl):.2f}, AP {np.exp(row.logHR_AP):.2f}")

KEY = [("p23407", "Total triglycerides"), ("p23408", "Triglycerides in VLDL"), ("p23454", "MUFA, % of total fatty acids"), ("p23406", "HDL cholesterol"),
       ("p23561", "Cholesterol in large HDL"), ("p23433", "Average diameter of HDL particles"), ("p23440", "Apolipoprotein A1"), ("p23439", "Apolipoprotein B"),
       ("p23470", "Glucose"), ("p23465", "Isoleucine"), ("p23466", "Leucine"), ("p23467", "Valine"), ("p23480", "Glycoprotein acetyls")]
rows = []
for c, name in KEY:
    for s, lab in [("all", "All matched sets"), ("biliary", "Biliary AP sets"), ("non-biliary", "Non-biliary AP sets")]:
        x = S[s].loc[c]
        rows.append([name, lab, hr(x.logHR_ctrl, x.se_ctrl), hr(x.logHR_AP, x.se_AP), pf(x.P_int), pf(x.FDR_int), pf(t3.loc[c, "P_3way"]) if s == "all" else "—"])
    P(f"KEY {name}: all ctrl {hr(a.loc[c,'logHR_ctrl'], a.loc[c,'se_ctrl'])} AP {hr(a.loc[c,'logHR_AP'], a.loc[c,'se_AP'])} P_int {pf(a.loc[c,'P_int'])} FDR {pf(a.loc[c,'FDR_int'])}; "
      f"biliary ctrl {hr(S['biliary'].loc[c,'logHR_ctrl'], S['biliary'].loc[c,'se_ctrl'])} AP {hr(S['biliary'].loc[c,'logHR_AP'], S['biliary'].loc[c,'se_AP'])} P_int {pf(S['biliary'].loc[c,'P_int'])}; "
      f"non-biliary ctrl {hr(S['non-biliary'].loc[c,'logHR_ctrl'], S['non-biliary'].loc[c,'se_ctrl'])} AP {hr(S['non-biliary'].loc[c,'logHR_AP'], S['non-biliary'].loc[c,'se_AP'])} P_int {pf(S['non-biliary'].loc[c,'P_int'])}; 3-way {pf(t3.loc[c,'P_3way'])}")
pd.DataFrame(rows, columns=["Measure", "Matched sets", "HR in controls (95% CI)", "HR in AP (95% CI)", "P for interaction", "FDR for interaction",
                            "P for etiology × AP × measure"]).to_csv(os.path.join(RES, "H03_tableS11.csv"), index=False)
rows = []
for s, lab in [("all", "Main analysis"), ("adjusted for fasting time", "Adjusted for time since last meal"),
               ("adjusted for cholesterol-lowering drugs", "Adjusted for cholesterol-lowering drugs"),
               ("no cholesterol-lowering drugs", "Excluding users of cholesterol-lowering drugs"),
               ("biliary, no cholesterol-lowering drugs", "Biliary AP sets, excluding users of cholesterol-lowering drugs"),
               ("inverse normal transformation", "Rank-based inverse normal transformation")]:
    x = S[s]; da = (a.logHR_AP - a.logHR_ctrl)[sig]; dx = (x.logHR_AP - x.logHR_ctrl)[sig]
    tg, hdl = x.loc["p23407"], x.loc["p23561"]
    rows.append([lab, f"{int(x.n_AP.iloc[0]):,}/{int(x.ev_AP.iloc[0])}", f"{int(x.n.iloc[0] - x.n_AP.iloc[0]):,}/{int(x.ev_ctrl.iloc[0])}",
                 int((x.FDR_ctrl < .05).sum()), int((x.FDR_int < .05).sum()), int((x.P_int < .05).sum()),
                 "—" if s == "all" else f"{(np.sign(da) == np.sign(dx)).mean()*100:.0f}",
                 f"{hr(tg.logHR_ctrl, tg.se_ctrl)} vs {hr(tg.logHR_AP, tg.se_AP)}", f"{hr(hdl.logHR_ctrl, hdl.se_ctrl)} vs {hr(hdl.logHR_AP, hdl.se_AP)}"])
pd.DataFrame(rows, columns=["Analysis", "AP, n/events", "Controls, n/events", "Measures associated with diabetes in controls, FDR <0.05",
                            "Interactions, FDR <0.05", "Interactions, P <0.05", "Same direction as main analysis, % of 98",
                            "Total triglycerides: HR in controls vs AP", "Cholesterol in large HDL: HR in controls vs AP"]).to_csv(os.path.join(RES, "H03_tableS12.csv"), index=False)
# Additional file 2: all 249 measures in every analysis set
with pd.ExcelWriter(os.path.join(RES, "Additional_file_2_NMR.xlsx")) as xw:
    readme = pd.DataFrame({"Additional file 2": [
        "Associations of 249 pre-attack NMR metabolic measures with incident diabetes in matched controls and in patients with acute pancreatitis (AP).",
        "HRs are per SD of the log-transformed, spectrometer-centred measure, from Cox models stratified by matched set with AP, BMI, HbA1c, the measure and AP × measure.",
        "HR_controls = exp(b_measure); HR_AP = exp(b_measure + b_interaction). FDR, Benjamini–Hochberg false discovery rate within each analysis.",
        "Sheets: 2a main analysis; 2b biliary AP matched sets; 2c non-biliary AP matched sets; 2d test of etiology (biliary vs non-biliary) × AP × measure; "
        "2e adjusted for time since last meal; 2f adjusted for cholesterol-lowering drugs; 2g excluding users of cholesterol-lowering drugs; "
        "2h biliary AP sets excluding users of cholesterol-lowering drugs; 2i rank-based inverse normal transformation."]})
    readme.to_excel(xw, sheet_name="README", index=False)
    SHEETS = [("all", "2a Main"), ("biliary", "2b Biliary AP"), ("non-biliary", "2c Non-biliary AP"), ("3-way", "2d Etiology test"),
              ("adjusted for fasting time", "2e Fasting-adjusted"), ("adjusted for cholesterol-lowering drugs", "2f Drug-adjusted"),
              ("no cholesterol-lowering drugs", "2g No lipid drugs"), ("biliary, no cholesterol-lowering drugs", "2h Biliary no lipid drugs"),
              ("inverse normal transformation", "2i Inverse normal")]
    for s, sheet in SHEETS:
        if s == "3-way":
            t = t3.reset_index(); t.insert(1, "measure", t.code.map(lambda c: title.get(c, c))); t.columns = ["UKB field", "Measure", "P for etiology × AP × measure", "FDR"]
            t.to_excel(xw, sheet_name=sheet, index=False); continue
        x = S[s]
        y = x.reset_index()
        y.insert(1, "measure", y.code.map(lambda c: title.get(c, c)))
        y["HR_controls"] = [hr(b_, s_) for b_, s_ in zip(y.logHR_ctrl, y.se_ctrl)]; y["HR_AP"] = [hr(b_, s_) for b_, s_ in zip(y.logHR_AP, y.se_AP)]
        y = y[["code", "measure", "n", "n_AP", "ev_AP", "ev_ctrl", "HR_controls", "P_ctrl", "FDR_ctrl", "HR_AP", "P_AP", "FDR_AP", "P_int", "FDR_int"]]
        y.columns = ["UKB field", "Measure", "Participants", "AP patients", "Events, AP", "Events, controls", "HR in controls (95% CI)", "P, controls", "FDR, controls",
                     "HR in AP (95% CI)", "P, AP", "FDR, AP", "P for interaction", "FDR for interaction"]
        y.to_excel(xw, sheet_name=sheet, index=False)
open(os.path.join(RES, "H03_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
