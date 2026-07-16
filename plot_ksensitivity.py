#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ksensitivity.py  -  K-sweep figure (R3#8 + R2#4), styled to Nature/Springer
figure conventions: sans-serif (Helvetica/Arial; falls back to Liberation Sans
then DejaVu Sans), sentence-case labels, 7-pt body text, 8-pt bold lowercase
panel labels, thin 0.5 spines, no grid, frame-less legends, the manuscript's
house teal-blue palette (+ one red accent) with distinct markers, 600 dpi.

Two panels:
  (a) candidate coverage rises with K (52%->100%), but recovery of the TRUE
      origin distribution (Top-1 / Recall@k vs the fixed full-544 target) stays
      flat -> omitted origins are unpredictable long-tail sources.
  (b) within-K (native) KL/Recall shift with K only because the target is
      renormalised over more candidates (an artifact, not a real change).

Usage:
    python plot_ksensitivity.py \
        --csv /home/lp43319/projects/GNN/visitgnn/output/KS_w1w2/ks_table.csv \
        --out /home/lp43319/projects/GNN/visitgnn/output/KS_w1w2/ksensitivity \
        --highlight_k 50
Writes <out>.png and <out>.pdf. --titles adds in-figure titles (working version);
--font_path /path/Helvetica.ttf registers a specific font if you have one.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FixedLocator

# ----- manuscript house palette (matches plot.py / evaluate_and_plot.py) -----
C_COVER = "#7F8A9B"   # blue-gray (context: coverage; dashed)
C_TOP1  = "#599CB4"   # medium teal-blue (primary)
C_REC   = "#B83945"   # house red accent (the headline flat line)
C_KL    = "#7A8CA1"   # blue-gray (native KL)
C_NDCG  = "#AECFD4"   # pale blue
C_DARK  = "#2F3B44"   # dark slate (annotations / vline)


def _band(ax, x, m, s, **kw):
    line, = ax.plot(x, m, **kw)
    if s is not None and np.any(np.asarray(s) > 0):
        m = np.asarray(m, float); s = np.asarray(s, float)
        ax.fill_between(x, m - s, m + s, color=line.get_color(), alpha=0.16, lw=0)
    return line


def _setup_font(font_path=None):
    """Helvetica/Arial first, with robust fallbacks; register a TTF if given."""
    chain = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        chain = [fm.FontProperties(fname=font_path).get_name()] + chain
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chain
    plt.rcParams["mathtext.fontset"] = "stixsans"   # sans-serif italics for $K$
    plt.rcParams["mathtext.default"] = "it"
    resolved = fm.findfont(fm.FontProperties(family=chain))
    return os.path.basename(resolved)


def main():
    ap = argparse.ArgumentParser(description="Plot the K-sensitivity table (Nature/Springer style).")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output path prefix (.png and .pdf)")
    ap.add_argument("--highlight_k", type=int, default=50)
    ap.add_argument("--recall_k", type=int, default=5)
    ap.add_argument("--ndcg_k", type=int, default=50)
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--titles", action="store_true", help="add in-figure titles (working version)")
    ap.add_argument("--font_path", default=None, help="optional .ttf to register (e.g. Helvetica)")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("K").reset_index(drop=True)
    K = df["K"].to_numpy()
    rk, nk = args.recall_k, args.ndcg_k
    has_full = "Top1_full" in df.columns and "coverage" in df.columns

    used_font = _setup_font(args.font_path)
    plt.rcParams.update({
        "font.size": 7,            # Nature: body text <= 7 pt
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.5,     # thin spines
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "figure.dpi": 300,
        "savefig.dpi": args.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    ncols = 2 if has_full else 1
    # single-column ~88 mm = 3.46 in; double-column ~180 mm = 7.1 in
    fig, axes = plt.subplots(1, ncols, figsize=(3.55 * ncols, 2.75), layout="constrained")
    axes = np.atleast_1d(axes)

    def _finish_x(ax):
        if args.logx:
            ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(K))
        ax.set_xticklabels([str(int(k)) for k in K])
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(False)
        ax.margins(x=0.03)

    def _tag(ax, t):
        ax.text(0.02, 0.985, t, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="left")

    leg_kw = dict(frameon=False, handlelength=1.5, labelspacing=0.35,
                  borderaxespad=0.4, handletextpad=0.5)

    # ---------------- Panel (a): coverage vs full-target recovery ----------------
    if has_full:
        ax = axes[0]
        _band(ax, K, df["coverage"], None, color=C_COVER, ls="--", marker="o",
              label="Coverage")
        _band(ax, K, df["Top1_full"], df.get("Top1_full_std"), color=C_TOP1,
              marker="s", label="Top-1 (full)")
        _band(ax, K, df[f"Recall@{rk}_full"], df.get(f"Recall@{rk}_full_std"),
              color=C_REC, marker="^", label=f"Recall@{rk} (full)")
        ax.axvline(args.highlight_k, color=C_DARK, ls=":", lw=0.7, alpha=0.7, zorder=0)
        ax.annotate(r"$K$ = 50", xy=(args.highlight_k, 0.30),
                    xytext=(args.highlight_k + 38, 0.30), fontsize=6.5, color=C_DARK,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=C_DARK, lw=0.6))
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(r"Candidate set size, $K$")
        ax.set_ylabel("Fraction")
        ax.legend(loc="center right", **leg_kw)
        _tag(ax, "(a)")
        if args.titles:
            ax.set_title("Coverage rises; recovery does not", fontsize=7)
        _finish_x(ax)

    # ---------------- Panel (b): native-@K artifact ----------------
    axB = axes[1] if has_full else axes[0]
    lN = _band(axB, K, df[f"Recall@{rk}"], df.get(f"Recall@{rk}_std"),
               color=C_TOP1, marker="^", label=f"Recall@{rk}")
    extra = []
    if f"NDCG@{nk}" in df.columns:
        lG = _band(axB, K, df[f"NDCG@{nk}"], df.get(f"NDCG@{nk}_std"),
                   color=C_NDCG, ls="-.", marker="o", label=f"NDCG@{nk}")
        extra.append(lG)
    axB.set_ylim(0, 1.03)
    axB.set_xlabel(r"Candidate set size, $K$")
    axB.set_ylabel("Native ranking metric")
    _finish_x(axB)

    axR = axB.twinx()
    axR.spines["right"].set_linewidth(0.5)
    lK = _band(axR, K, df["KL"], df.get("KL_std"), color=C_KL, marker="s", label="KL")
    axR.set_ylabel("Native KL (lower is better)", color=C_KL)
    axR.tick_params(axis="y", labelcolor=C_KL, width=0.5, length=2.5, direction="out")
    axR.grid(False)

    lines = [lN] + extra + [lK]
    axB.legend(lines, [ln.get_label() for ln in lines], loc="center right", **leg_kw)
    _tag(axB, "(b)")
    if args.titles:
        axB.set_title("Within-K metrics are renormalization artifacts", fontsize=7)

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"[plot] wrote {args.out}.png and {args.out}.pdf  (font={used_font}, dpi={args.dpi})")
    if used_font.lower().startswith(("dejavu",)):
        print("[plot] NOTE: fell back to DejaVu Sans (Arial/Helvetica/Liberation not found). "
              "Install 'fonts-liberation' or pass --font_path Arial.ttf for a true Arial look.")
    if has_full:
        r0, r1 = df[f"Recall@{rk}_full"].iloc[0], df[f"Recall@{rk}_full"].iloc[-1]
        t0, t1 = df["Top1_full"].iloc[0], df["Top1_full"].iloc[-1]
        c0, c1 = df["coverage"].iloc[0], df["coverage"].iloc[-1]
        print(f"[plot] coverage {c0:.0%}->{c1:.0%} (K={int(K[0])}->{int(K[-1])}); "
              f"Recall@{rk}_full {r0:.3f}->{r1:.3f}, Top1_full {t0:.3f}->{t1:.3f}.")


if __name__ == "__main__":
    main()
