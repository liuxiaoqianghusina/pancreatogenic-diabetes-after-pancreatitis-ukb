"""13: Figures 1–3 in Diabetes Care style.
Text 8–10 pt (Arial); panels labelled A, B, C… only; HbA1c in % (mmol/mol); colour is never the only cue (controls dashed,
excess hatched, comparisons expected to be null shown as circles); contrast ≥4.5:1."""
import re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter
from common import ROOT, RES, HLAB, os
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "hatch.linewidth": 0.6,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
C_AP, C_CT = "#B2342C", "#4F5B63"   # contrast against white about 5.7:1 and 7.0:1
FIG = os.path.join(ROOT, "figures"); R = lambda f: pd.read_csv(os.path.join(RES, f))
num = lambda s: [float(v) for v in re.findall(r"-?\d+\.?\d*", str(s).replace("−", "-"))]
HL2 = ["<5.4%\n(<36)", "5.4–5.6%\n(36–38)", "5.7–5.9%\n(39–41)", "6.0–6.4%\n(42–47)"]
HT = ["HbA1c <5.4% (<36 mmol/mol)", "HbA1c 5.4–5.6% (36–38 mmol/mol)", "HbA1c 5.7–5.9% (39–41 mmol/mol)", "HbA1c 6.0–6.4% (42–47 mmol/mol)"]
def letter(ax, s): ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
def save(fig, name):
    for e in ["png", "pdf", "tiff"]: fig.savefig(os.path.join(FIG, f"{name}.{e}"), dpi=600, **({"pil_kwargs": {"compression": "tiff_lzw"}} if e == "tiff" else {}))
    plt.close(fig)

# ---------------- Figure 1 ----------------
m = pd.read_pickle(os.path.join(ROOT, "cache", "matched_main.pkl"))
def aj(d):
    d = d.reset_index(drop=True); f = AalenJohansenFitter(calculate_variance=True, seed=1).fit(d["T"], pd.Series(d.E.values), event_of_interest=1)
    s = f.cumulative_density_.iloc[:, 0] * 100; c = f.confidence_interval_cumulative_density_ * 100
    return s.index.values, s.values, c.iloc[:, 0].values, c.iloc[:, 1].values
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
for k, (ax, h) in enumerate(zip(axes.flat, HLAB)):
    s = m[m.hcat == h]
    for e, col, ls, lab in [(1, C_AP, "-", "Acute pancreatitis"), (0, C_CT, "--", "Matched controls")]:
        t, y, lo, hi = aj(s[s.exp == e]); kk = t <= 10
        ax.step(t[kk], y[kk], where="post", color=col, lw=1.3, ls=ls, label=lab); ax.fill_between(t[kk], lo[kk], hi[kk], step="post", color=col, alpha=0.14, lw=0)
    ax.set_xlim(0, 10); ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_xlabel("Years after index date"); ax.set_ylabel("Cumulative incidence (%)")
    ax.text(0.03, 0.97, HT[k], transform=ax.transAxes, va="top", fontsize=8); letter(ax, "ABCD"[k])
h_, l_ = axes[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "DC_Fig1_CIF")

# ---------------- Figure 2 ----------------
af, sp, ps, jt = R("F06_strata_AF.csv"), R("F06_spline_HR.csv"), R("H01_prs_strata.csv"), R("H01_joint_hba1c_prs.csv")
H = ["HbA1c <36", "HbA1c 36-38", "HbA1c 39-41", "HbA1c 42-47"]; T = ["T1 low", "T2 intermediate", "T3 high"]; TL = ["Low", "Intermediate", "High"]
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.8))
def bars(a, ctrl, apv, rds, labels, xlabel, ymax):
    x = np.arange(len(labels))
    a.bar(x, ctrl, color=C_CT, width=0.6, label="Expected without pancreatitis (matched controls)")
    a.bar(x, np.array(apv) - np.array(ctrl), bottom=ctrl, color="white", edgecolor=C_AP, hatch="////", lw=0.9, width=0.6, label="Excess attributable to pancreatitis")
    for i in range(len(x)): a.text(i, apv[i] + ymax * 0.02, f"+{rds[i]:.1f}", ha="center", fontsize=8, color=C_AP)
    a.set_xticks(x); a.set_xticklabels(labels); a.set_xlabel(xlabel); a.set_ylabel("5-year cumulative incidence (%)"); a.set_ylim(0, ymax)
r = af.set_index("group").loc[H]
bars(ax[0, 0], r.CIF5_ctrl.values, r.CIF5_AP.values, [num(v)[0] for v in r.risk_difference], HL2, "Pre-attack HbA1c, % (mmol/mol)", 40)
ax[0, 0].legend(frameon=False, loc="upper left"); letter(ax[0, 0], "A")
a = ax[0, 1]; a.fill_between(sp.hba1c, sp.lo, sp.hi, color=C_AP, alpha=0.14, lw=0); a.plot(sp.hba1c, sp.HR, color=C_AP, lw=1.3)
for xm, h in zip([33, 37.5, 40.5, 44.5], H):
    e, lo, hi = num(r.loc[h, "HR"])[:3]; a.errorbar(xm, e, yerr=[[e - lo], [hi - e]], fmt="o", color="black", ms=3.5, lw=0.9, capsize=2)
a.axhline(1, color="grey", lw=0.7, ls=":"); a.set_yscale("log"); a.set_yticks([0.5, 1, 2, 4, 8, 16]); a.set_yticklabels(["0.5", "1", "2", "4", "8", "16"])
xt = [31, 36, 42, 47]; a.set_xticks(xt); a.set_xticklabels([f"{v/10.929+2.15:.1f}\n({v})" for v in xt])
a.set_xlabel("Pre-attack HbA1c, % (mmol/mol)"); a.set_ylabel("HR for diabetes (AP vs. controls)"); letter(a, "B")
q = ps.set_index("PRS_tertile").loc[T]
bars(ax[1, 0], q.CIF5_ctrl.values, q.CIF5_AP.values, [num(v)[0] for v in q.risk_difference], TL, "Type 2 diabetes polygenic risk (tertile)", 14); letter(ax[1, 0], "C")
a = ax[1, 1]
items = [(f"PRS {l.lower()}", q.HR.iloc[i]) for i, l in enumerate(TL)] + [(f"HbA1c <5.7%, PRS {l.lower()}", jt[(jt.HbA1c == "HbA1c <39") & (jt.PRS == T[i])].HR.iloc[0]) for i, l in enumerate(TL)]
yy = np.arange(len(items))[::-1]
for y_, (lab, v) in zip(yy, items):
    e, lo, hi = num(v)[:3]; a.plot([lo, hi], [y_, y_], color="black", lw=0.9); a.plot(e, y_, "s", color=C_AP, ms=4)
a.set_yticks(yy); a.set_yticklabels([i[0] for i in items]); a.axvline(1, color="grey", lw=0.7, ls=":")
a.set_xscale("log"); a.set_xticks([1, 2, 4, 8, 16]); a.set_xticklabels(["1", "2", "4", "8", "16"]); a.set_xlim(0.8, 16)
a.set_xlabel("HR for diabetes (AP vs. controls)"); letter(a, "D")
fig.tight_layout(); fig.subplots_adjust(hspace=0.55, wspace=0.85); save(fig, "DC_Fig2_HbA1c_PRS")

# ---------------- Figure 3 ----------------
act, sens, nc, fg = R("F02_active.csv"), R("F03_sensitivity.csv"), R("F03_negative_control.csv"), R("F04_finegray.csv")
tv, mi, sev, pop, gp = R("G07_1a_timevarying.csv"), R("G07_1b_mice.csv"), R("G07_2_severity_interventions.csv"), R("F06_comparator_vs_population.csv"), R("F11_gp_linked.csv")
pick = lambda d, col, key, keycol: d[d[keycol] == key][col].iloc[0]
items = [("Main analysis", None),
         ("HbA1c- and BMI-matched controls", pick(sens, "HR_all", "Main analysis", "analysis")),
         ("Multiple imputation of covariates", mi.HR.iloc[0]),
         ("Full cohort, time-varying exposure", pick(tv, "HR", "Fully adjusted", "model")),
         ("Fine–Gray model (death as competing risk)", fg.SHR.iloc[0]),
         ("Additionally matched on polygenic risk", pick(ps, "HR", "All", "PRS_tertile")),
         ("Severity of pancreatitis", None),
         ("No organ failure, pseudocyst, or intervention", pick(sev, "HR", "None of these markers", "group")),
         ("Organ failure or pseudocyst", pick(sev, "HR", "Organ failure or pseudocyst (no intervention)", "group")),
         ("Necrosectomy or drainage", pick(sev, "HR", "Pancreatic intervention (necrosectomy/drainage)", "group")),
         ("Active comparators", None),
         ("AP vs. acute appendicitis", pick(act, "HR", "vs appendicitis | fully adjusted", "comparison")),
         ("AP vs. acute cholecystitis", pick(act, "HR", "vs cholecystitis | fully adjusted", "comparison")),
         ("AP vs. both, HbA1c <5.4%", pick(act, "HR", "vs both | HbA1c <36 | fully adjusted", "comparison")),
         ("Biliary AP vs. acute cholecystitis", pick(act, "HR", "biliary AP vs cholecystitis | fully adjusted + cholecystectomy", "comparison")),
         ("Appendicitis vs. matched population*", pop.HR.iloc[0]),
         ("Sensitivity analyses", None),
         ("Hospital records only", pick(sens, "HR_all", "S1 AP and diabetes from hospital records (HES) only", "analysis")),
         ("AP ≤3 years after HbA1c measurement", pick(sens, "HR_all", "S4 AP within 3 years of HbA1c measurement", "analysis")),
         ("180-day blanking period", pick(sens, "HR_all", "S5 180-day blanking period", "analysis")),
         ("Censoring at chronic pancreatitis", pick(sens, "HR_all", "S7 Censoring at chronic pancreatitis", "analysis")),
         ("Primary-care–linked participants", gp.HR.iloc[0]),
         ("Negative-control outcomes*", None),
         ("Senile cataract", pick(nc, "HR", "Senile cataract H25", "outcome")),
         ("Knee osteoarthritis", pick(nc, "HR", "Knee osteoarthritis M17", "outcome"))]
fig, ax = plt.subplots(figsize=(7.0, 6.6)); y = len(items); sec = ""
for lab, v in items:
    y -= 1
    if v is None:
        sec = lab; ax.text(-1.40, y, lab, transform=ax.get_yaxis_transform(), fontweight="bold", va="center"); continue
    e, lo, hi = num(v)[:3]; null = sec.startswith("Negative") or lab.endswith("*")
    ax.plot([max(lo, 0.5), min(hi, 64)], [y, y], color="black", lw=0.9)
    ax.plot(e, y, "o" if null else "s", color=C_CT if null else C_AP, ms=4.5 if null else 4)
    if hi > 64: ax.annotate("", xy=(64, y), xytext=(46, y), arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(-1.34, y, lab, transform=ax.get_yaxis_transform(), va="center")
    ax.text(1.03, y, f"{e:.2f} ({lo:.2f}–{hi:.2f})", transform=ax.get_yaxis_transform(), va="center")
ax.axvline(1, color="grey", lw=0.7, ls=":"); ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64"])
ax.set_xlim(0.5, 64); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_ylim(-0.8, len(items) - 0.2)
ax.set_xlabel("HR for incident diabetes (95% CI)")
fig.subplots_adjust(left=0.47, right=0.79, top=0.99, bottom=0.08); save(fig, "DC_Fig3_forest")
print("ok")
