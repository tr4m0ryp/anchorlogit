"""Figures for the note (vector PDF, column width, Okabe-Ito colours).

    python paper/figures/make.py [--results DIR] [--raw DIR] [--out DIR]

fig1_paired.pdf       the claim: paired test log-loss difference (ours - arm) with 95% CI
                      (t quantile x the benchmark's between-repeat SE) per dataset and n_train
fig2_reliability.pdf  calibration: reliability diagrams on pooled test rows (15 equal-mass bins),
                      ECE in the legend = the benchmark's ece15 (mean over repeats)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import expit  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def ci95(p):
    return stats.t.ppf(0.975, p["repeats"] - 1) * p["se"]


def forest(ax, res, cells, comps, title, xlabel):
    """One row per n_train, one marker per comparison, 95% CI bars."""
    k = len(comps)
    for i, n in enumerate(cells):
        for j, (key, label, col, mk) in enumerate(comps):
            p = res["by_n_train"][n]["paired"][key]["logloss"]
            y = i + (j - (k - 1) / 2) * 0.17
            ax.errorbar(p["mean"], y, xerr=ci95(p), fmt=mk, color=col, ms=3.5, lw=1,
                        capsize=1.5, label=label if i == 0 else None)
    ax.axvline(0, color="0.5", lw=0.7, ls="--")
    ax.set_yticks(range(len(cells)), [f"n = {int(n):,}" for n in cells])
    ax.set_ylim(-0.6, len(cells) - 0.4)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def fig1(R, out):
    main = [("a6_offset_kcv - a1_plain", "vs no reference", OI["verm"], "o"),
            ("a6_offset_kcv - a3_ref_feature", "vs reference as std. feature", OI["orange"], "s"),
            ("a6_offset_kcv - a4_ref_feature_x100", "vs reference as free column", OI["blue"], "D"),
            ("a6_offset_kcv - a5_offset_k1", "vs full offset ($\\kappa$ = 1)", OI["green"], "^")]
    rule = [("v_kappa_1se - a4_ref_feature_x100", "vs free column (CV C)", OI["blue"], "D"),
            ("v_kappa_1se - v_free_1se", "vs free column (same C rule)", OI["sky"], "v"),
            ("v_kappa_1se - a2_ref", "vs reference alone", OI["purple"], "P")]
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.4), constrained_layout=True)
    xl = "$\\Delta$ test log-loss (nats), ours $-$ arm"
    forest(axs[0, 0], R["higgs"], ["250", "500", "1000", "2000"], main,
           "(a) HIGGS: offset, $\\kappa$ by CV", xl)
    forest(axs[0, 1], R["bank"], ["250", "1000"], main, "(b) Bank: offset, $\\kappa$ by CV", xl)
    forest(axs[1, 0], R["elec2"], ["250", "1000"], main,
           "(c) Elec2 (aged reference): offset, $\\kappa$ by CV", xl)
    axs[1, 0].set_xscale("symlog", linthresh=0.01)
    forest(axs[1, 1], R["higgs_smalln_1se_controls"], ["100", "150", "250", "1000"], rule,
           "(d) HIGGS: one-SE rule toward $\\kappa$ = 1", xl)
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    axs[0, 1].legend(h1, l1, loc="lower right", frameon=False)
    axs[1, 1].legend(loc="lower right", frameon=False)
    fig.savefig(out / "fig1_paired.pdf")
    fig.savefig(out / "fig1_paired.png", dpi=200)
    plt.close(fig)


def load_raw(path, n, arms):
    z = np.load(path)
    reps = sorted({int(m.group(1)) for k in z.files if (m := re.match(rf"n{n}_r(\d+)_y$", k))})
    y = np.concatenate([z[f"n{n}_r{r}_y"] for r in reps]).astype(float)
    return y, {a: expit(np.concatenate([z[f"n{n}_r{r}_{a}"] for r in reps]).astype(float))
               for a in arms}


def reliability(ax, y, probs, ece, arms, title):
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
    for a, (label, col, mk) in arms.items():
        p = probs[a]
        order = np.argsort(p, kind="stable")
        bins = np.array_split(order, 15)
        px = [p[b].mean() for b in bins]
        py = [y[b].mean() for b in bins]
        ax.plot(px, py, marker=mk, ms=3, lw=1, color=col, label=f"{label}, ECE {ece[a]:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted probability (bin mean)")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)


def fig2(R, raw, out):
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.9), constrained_layout=True)
    arms = {"a1_plain": ("no reference", OI["verm"], "o"),
            "a3_ref_feature": ("ref. as std. feature", OI["orange"], "s"),
            "a4_ref_feature_x100": ("ref. as free column", OI["blue"], "D"),
            "a6_offset_kcv": ("offset, $\\kappa$ by CV", OI["black"], "x")}
    y, pr = load_raw(raw / "higgs_test_logodds.npz", 250, arms)
    ece = {a: R["higgs"]["by_n_train"]["250"]["arms"][a]["ece15"] for a in arms}
    reliability(axs[0], y, pr, ece, arms, "(a) HIGGS, n = 250")
    arms = {"a2_ref": ("reference alone", OI["purple"], "P"),
            "a5_offset_k1": ("full offset, $\\kappa$ = 1", OI["green"], "^"),
            "a4_ref_feature_x100": ("ref. as free column", OI["blue"], "D"),
            "a6_offset_kcv": ("offset, $\\kappa$ by CV", OI["black"], "x")}
    y, pr = load_raw(raw / "bank_shiftref_test_logodds.npz", 1000, arms)
    ece = {a: R["bank_shiftref"]["by_n_train"]["1000"]["arms"][a]["ece15"] for a in arms}
    reliability(axs[1], y, pr, ece, arms, "(b) Bank, shifted reference, n = 1,000")
    fig.savefig(out / "fig2_reliability.pdf")
    fig.savefig(out / "fig2_reliability.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "bench" / "results"))
    default = os.environ.get("ANCHORLOGIT_BENCH_RAW", str(REPO.parent / "_data" / "anchorlogit_bench"))
    ap.add_argument("--raw", default=default)
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    R = {p.stem: json.loads(p.read_text()) for p in Path(a.results).glob("*.json")}
    out = Path(a.out)
    fig1(R, out)
    fig2(R, Path(a.raw), out)
    print("wrote", *sorted(x.name for x in out.glob("fig*.pdf")))


if __name__ == "__main__":
    main()
