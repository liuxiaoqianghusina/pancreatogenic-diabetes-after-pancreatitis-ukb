"""19: Figure 4. Pre-attack NMR metabolic profile and diabetes in matched controls
vs patients with acute pancreatitis.
A: HR per SD of each of the 249 measures in controls (x) vs in patients with AP (y); filled points, AP x measure interaction FDR<0.05.
B, C: selected measures in biliary and non-biliary matched sets; controls open squares, AP filled circles.
Text 8 pt Arial; colour is never the only cue."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RES
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
C_AP, C_CT = "#B2342C", "#4F5B63"
FIG = os.path.join(ROOT, "figures")
r = pd.read_csv(os.path.join(RES, "H03_nmr_interaction.csv"))
a = r[r.set == "all"].copy()
KEY = [("p23407", "Total triglycerides"), ("p23408", "VLDL triglycerides"), ("p23454", "MUFA, % of fatty acids"), ("p23406", "HDL cholesterol"),
       ("p23561", "Cholesterol in large HDL"), ("p23433", "HDL particle diameter"), ("p23470", "Glucose"), ("p23467", "Valine")]
LAB_A = {"p23407": "Total triglycerides", "p23406": "HDL cholesterol", "p23561": "Cholesterol in large HDL", "p23454": "MUFA %"}


def letter(ax, s, x=-0.16):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


fig = plt.figure(figsize=(6.7, 6.8))   # saved at full size (no tight bbox): 17.0 cm wide
gsA = fig.add_gridspec(1, 1, left=0.12, right=0.98, top=0.92, bottom=0.57)
gsB = fig.add_gridspec(1, 2, wspace=0.10, left=0.29, right=0.98, top=0.43, bottom=0.12)
# ---- A: scatter ----
ax = fig.add_subplot(gsA[0, 0])
x, y = np.exp(a.logHR_ctrl), np.exp(a.logHR_AP); sig = a.FDR_int < 0.05
lim = (0.5, 2.2); ylim = (0.55, 1.8)
inside = x.between(*lim) & y.between(*ylim)
for _, row in a[~inside].iterrows(): print("off scale in panel A:", row.code, round(np.exp(row.logHR_ctrl), 2), round(np.exp(row.logHR_AP), 2))
x, y, sig = x[inside], y[inside], sig[inside]
ax.axhline(1, color="grey", lw=0.7, ls=":"); ax.axvline(1, color="grey", lw=0.7, ls=":")
ax.plot(lim, lim, color="grey", lw=0.8, ls="--", label="Equal association in both groups")
ax.scatter(x[~sig], y[~sig], s=14, facecolors="white", edgecolors=C_CT, linewidths=0.7, label="Interaction FDR ≥0.05", zorder=3)
ax.scatter(x[sig], y[sig], s=16, marker="^", color=C_AP, edgecolors="none", label="Interaction FDR <0.05", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*ylim)
t = [0.5, 0.67, 1, 1.5, 2]; ty = [0.67, 1, 1.5]
ax.set_xticks(t); ax.set_xticklabels([str(v) for v in t]); ax.set_yticks(ty); ax.set_yticklabels([str(v) for v in ty]); ax.minorticks_off()
ax.set_xlabel("HR for diabetes per SD in matched controls"); ax.set_ylabel("HR for diabetes per SD\nin patients with AP")
offs = {"p23407": (1.25, 0.68), "p23406": (0.86, 0.63), "p23561": (0.6, 1.45), "p23454": (1.75, 0.78)}
for c, lab in LAB_A.items():
    row = a[a.code == c].iloc[0]; px, py = np.exp(row.logHR_ctrl), np.exp(row.logHR_AP)
    ax.annotate(lab, xy=(px, py), xytext=offs[c], fontsize=8, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black", shrinkA=1, shrinkB=2))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.4, columnspacing=1.5)
letter(ax, "A", x=-0.13)
# ---- B, C: forests ----
f = lambda b, s: (np.exp(b), np.exp(b - 1.96 * s), np.exp(b + 1.96 * s))
for k, (setname, lab) in enumerate([("biliary", "B"), ("non-biliary", "C")]):
    ax = fig.add_subplot(gsB[0, k]); s = r[r.set == setname].set_index("code")
    ys = np.arange(len(KEY))[::-1]
    for yy, (c, name) in zip(ys, KEY):
        for grp, dy, col, mk, mfc in [("ctrl", 0.17, C_CT, "s", "white"), ("AP", -0.17, C_AP, "o", C_AP)]:
            e, lo, hi = f(s.loc[c, f"logHR_{grp}"], s.loc[c, f"se_{grp}"])
            ax.plot([lo, hi], [yy + dy, yy + dy], color=col, lw=0.9)
            ax.plot(e, yy + dy, mk, color=col, mfc=mfc, ms=4, mew=0.9)
    ax.axvline(1, color="grey", lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_xlim(0.45, 2.2); ax.set_xticks([0.5, 0.75, 1, 1.5, 2]); ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2"]); ax.minorticks_off()
    ax.set_ylim(-0.7, len(KEY) - 0.3); ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n in KEY] if k == 0 else [])
    if k == 1: ax.tick_params(axis="y", length=0)
    ax.set_xlabel("HR for diabetes per SD (95% CI)")
    ax.set_title("Biliary AP matched sets" if setname == "biliary" else "Non-biliary AP matched sets", fontsize=8, loc="left")
    letter(ax, lab, x=-0.72 if k == 0 else -0.06)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=C_CT, marker="s", mfc="white", lw=0.9, ms=4), Line2D([], [], color=C_AP, marker="o", lw=0.9, ms=4)]
fig.legend(h, ["Matched controls", "Patients with AP"], frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.62, 0.0))
for e in ["png", "pdf", "tiff"]:
    fig.savefig(os.path.join(FIG, f"CD_Fig4_metabolic.{e}"), dpi=600,
                **({"pil_kwargs": {"compression": "tiff_lzw"}} if e == "tiff" else {}))
print("ok")
