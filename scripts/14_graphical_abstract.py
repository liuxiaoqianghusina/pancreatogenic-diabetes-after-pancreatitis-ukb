"""14: Graphical abstract (landscape, three panels: design → findings → implications). Stratum estimates are read from results/;
cohort counts in the design panel are those reported in the paper. Prints any text that overruns its panel."""
import re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import ROOT, RES, os
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "hatch.linewidth": 0.6,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
C_AP, C_CT, C_HEAD, C_BG = "#B2342C", "#4F5B63", "#1F3B57", "#F4F6F8"
R = lambda f: pd.read_csv(os.path.join(RES, f)); num = lambda s: [float(v) for v in re.findall(r"-?\d+\.?\d*", str(s).replace("−", "-"))]
af = R("F06_strata_AF.csv").set_index("group").loc[["HbA1c <36", "HbA1c 36-38", "HbA1c 39-41", "HbA1c 42-47"]]
sev = R("G07_2_severity_interventions.csv"); act = R("F02_active.csv"); ps = R("H01_prs_strata.csv").set_index("PRS_tertile")

fig = plt.figure(figsize=(9.0, 4.6)); W, H = 9.0, 4.6
base = fig.add_axes([0, 0, 1, 1]); base.set_xlim(0, W); base.set_ylim(0, H); base.axis("off")
base.text(W / 2, H - 0.32, "Diabetes after acute pancreatitis is largely pancreatogenic", ha="center", va="center", fontsize=13, fontweight="bold", color=C_HEAD)
cols = [(0.20, 2.55, "Design"), (3.05, 3.30, "Findings"), (6.65, 2.15, "Implications")]
for x0, w, t in cols:
    base.add_patch(FancyBboxPatch((x0, 0.25), w, H - 1.0, boxstyle="round,pad=0.02,rounding_size=0.12", fc=C_BG, ec="#C9D1D8", lw=0.8))
    base.text(x0 + w / 2, H - 0.92, t, ha="center", va="center", fontsize=11, fontweight="bold", color=C_HEAD)
for xa, xb in [(2.78, 3.02), (6.38, 6.62)]:
    base.add_patch(FancyArrowPatch((xa, 1.95), (xb, 1.95), arrowstyle="-|>", mutation_scale=14, color=C_HEAD, lw=1.4))

# ---- design panel ----
dl = [("UK Biobank", "432,484 adults without diabetes"),
      ("Pre-attack HbA1c", "measured a median 7.5 years\nbefore pancreatitis"),
      ("2,072 incident AP", "vs. 10,360 controls matched on\nage, sex, pre-attack HbA1c and BMI"),
      ("Triangulation", "type 2 diabetes polygenic risk,\nappendicitis and cholecystitis\nas active comparators")]
y = H - 1.32
for head, body in dl:
    base.text(0.38, y, head, fontsize=9, fontweight="bold", color=C_HEAD, va="top")
    base.text(0.38, y - 0.21, body, fontsize=8.5, color="#222222", va="top", linespacing=1.25)
    y -= 0.27 + 0.20 * (body.count("\n") + 1) + 0.12

# ---- findings panel: stacked bars ----
ax = fig.add_axes([3.55 / W, 1.62 / H, 1.30 / W, 1.50 / H])
x = np.arange(4); ct, apv = af.CIF5_ctrl.values, af.CIF5_AP.values
ax.bar(x, ct, color=C_CT, width=0.62); ax.bar(x, apv - ct, bottom=ct, color="white", edgecolor=C_AP, hatch="////", lw=0.8, width=0.62)
for i, r in enumerate(af.risk_difference): ax.text(i, apv[i] + 0.8, f"+{num(r)[0]:.1f}", ha="center", fontsize=8, color=C_AP)
ax.set_xticks(x); ax.set_xticklabels(["<5.4", "5.4–\n5.6", "5.7–\n5.9", "6.0–\n6.4"], fontsize=8); ax.tick_params(axis="y", labelsize=8)
ax.set_ylim(0, 31); ax.set_ylabel("5-year diabetes (%)", fontsize=8); ax.set_xlabel("Pre-attack HbA1c (%)", fontsize=8)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
from matplotlib.patches import Rectangle
base.add_patch(Rectangle((3.32, 0.98), 0.16, 0.11, fc=C_CT, ec=C_CT)); base.text(3.54, 1.035, "Expected (controls)", fontsize=8, va="center")
base.add_patch(Rectangle((3.32, 0.78), 0.16, 0.11, fc="white", ec=C_AP, hatch="////", lw=0.8)); base.text(3.54, 0.835, "Excess due to AP", fontsize=8, va="center")
base.text(4.70, 0.47, "≈5 extra cases per 100 in 5 years, whatever the\npre-attack HbA1c or polygenic risk", ha="center", fontsize=8.5, color=C_AP, va="center", fontweight="bold", linespacing=1.2)
# ---- findings panel: key estimates ----
lines = [("By pre-attack HbA1c", f"{num(af.HR.iloc[0])[0]:.1f} → {num(af.HR.iloc[3])[0]:.1f}"),
         ("By polygenic risk", f"{num(ps.loc['T1 low','HR'])[0]:.1f} → {num(ps.loc['T3 high','HR'])[0]:.1f}"),
         ("Versus appendicitis", f"HR {num(act[act.comparison=='vs appendicitis | fully adjusted'].HR.iloc[0])[0]:.1f}"),
         ("By severity", f"{num(sev.HR.iloc[0])[0]:.1f} → {num(sev.HR.iloc[1])[0]:.1f} → {num(sev.HR.iloc[2])[0]:.0f}")]
yy = H - 1.30
for a, b in lines:
    base.text(4.98, yy, a, fontsize=8.5, color="#222222", va="top"); base.text(4.98, yy - 0.19, b, fontsize=9, color=C_AP, va="top", fontweight="bold")
    yy -= 0.50
base.text(4.98, yy + 0.02, "Pre-attack HbA1c no\nhigher than in peers", fontsize=8.5, color="#222222", va="top", linespacing=1.2)

# ---- implications panel ----
il = [("Mechanism", "pancreatic injury, not unmasked\nmetabolic or genetic risk"),
      ("Who", "all survivors, including those with\nnormal pre-attack glycemia"),
      ("When", "long term: median diagnosis\n3.7 years after the attack"),
      ("Priority", "prediabetes before the attack:\n~1 in 4 develop diabetes in 5 years")]
y = H - 1.32
for head, body in il:
    base.text(6.83, y, head, fontsize=9, fontweight="bold", color=C_HEAD, va="top")
    base.text(6.83, y - 0.21, body, fontsize=8.5, color="#222222", va="top", linespacing=1.25)
    y -= 0.27 + 0.20 * (body.count("\n") + 1) + 0.14
base.text(W / 2, 0.08, "AP, acute pancreatitis; HR (hazard ratio) of AP vs. matched controls, shown from lowest to highest stratum.", ha="center", va="bottom", fontsize=7.5, color="#555555")
fig.canvas.draw(); rr = fig.canvas.get_renderer(); bad = []
for t in base.texts:
    bb = t.get_window_extent(rr).transformed(base.transData.inverted())
    for x0, w, name in cols:
        if x0 - 0.05 <= bb.x0 <= x0 + w and bb.y0 > 0.25 and bb.y1 < H - 0.75 and bb.x1 > x0 + w - 0.06:
            bad.append((name, t.get_text()[:30], round(bb.x1, 2), round(x0 + w, 2)))
print("Text overrunning a panel:", bad if bad else "none")
for e in ["png", "pdf", "tiff"]:
    fig.savefig(os.path.join(ROOT, "figures", f"DC_graphical_abstract.{e}"), dpi=600, **({"pil_kwargs": {"compression": "tiff_lzw"}} if e == "tiff" else {}))
print("ok")
