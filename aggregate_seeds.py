#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_seeds.py   (R3#10)
----------------------------
Aggregate multi-seed results for the leaky (w1->w1) and leakage-free (w1->w2)
models. Metrics are computed on the COMMON valid-POI intersection (correct
denominator -- the same fix as eval_on_intersection.py), then reported as
mean +/- std across seeds, with a paired (by training seed) significance test
of the leakage effect (free - leaky).

Reuses the exact metric definitions from eval_on_intersection.py, so numbers are
directly comparable to the single-run intersection table.

Usage:
  python aggregate_seeds.py \
    --leaky_glob "/home/lp43319/projects/GNN/visitgnn/output/MS_w1w1/seed_*/prediction/*_preds_test_with_gt.csv" \
    --free_glob  "/home/lp43319/projects/GNN/visitgnn/output/MS_w1w2/seed_*/prediction/*_preds_test_with_gt.csv" \
    --ndcg_k 50 --recall_k 5

(Quote the globs so the shell does not expand them.)
"""
import argparse
import glob
import re
import numpy as np

from eval_on_intersection import load_with_gt, metrics_on, EPS


def seed_of(path):
    m = re.search(r"seed[_-]?(\d+)", path)
    return int(m.group(1)) if m else None


def load_set(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match: {pattern}")
    out = {}
    for f in files:
        s = seed_of(f)
        if s is None:
            raise ValueError(f"cannot parse seed from path (expected .../seed_<n>/...): {f}")
        out[s] = load_with_gt(f)
    return out, files


def mean_std(xs):
    a = np.asarray(xs, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate multi-seed leaky vs leakage-free on the common valid-POI intersection.")
    ap.add_argument("--leaky_glob", required=True, help="glob for leaky (w1->w1) *_preds_test_with_gt.csv")
    ap.add_argument("--free_glob", required=True, help="glob for leakage-free (w1->w2) *_preds_test_with_gt.csv")
    ap.add_argument("--ndcg_k", type=int, default=50)
    ap.add_argument("--recall_k", type=int, default=5)
    ap.add_argument("--per_seed", action="store_true", help="also print the full per-seed table")
    args = ap.parse_args()

    leaky, lf = load_set(args.leaky_glob)
    free, ff = load_set(args.free_glob)
    seeds = sorted(set(leaky) & set(free))
    if not seeds:
        raise SystemExit("no common seeds between leaky and free sets")
    print(f"leaky seeds: {sorted(leaky)}  ({len(lf)} files)")
    print(f"free  seeds: {sorted(free)}  ({len(ff)} files)")
    print(f"paired seeds (used): {seeds}\n")

    # Common valid-POI intersection: POIs with target mass>0 in EVERY leaky AND EVERY free seed.
    def valid_set(store):
        return {p for p, (pr, gt) in store.items() if gt.sum() > EPS}
    inter = None
    for s in seeds:
        vs = valid_set(leaky[s]) & valid_set(free[s])
        inter = vs if inter is None else (inter & vs)
    inter = sorted(inter)
    print(f"common valid-POI intersection across all seeds: {len(inter)} POIs\n")
    if not inter:
        raise SystemExit("empty intersection; check the two graphs share the same POI sampling.")

    keys = ['KL', 'MAE', 'Top1', f'NDCG@{args.ndcg_k}', f'Recall@{args.recall_k}']
    Lm = {k: [] for k in keys}
    Fm = {k: [] for k in keys}
    for s in seeds:
        mL = metrics_on(inter, leaky[s], args.ndcg_k, args.recall_k)
        mF = metrics_on(inter, free[s], args.ndcg_k, args.recall_k)
        for k in keys:
            Lm[k].append(mL[k])
            Fm[k].append(mF[k])

    if args.per_seed:
        print("=== Per-seed (intersection) ===")
        hdr = "seed  " + "".join(f"{('L_'+k):>14}{('F_'+k):>14}" for k in ['KL', 'Top1'])
        print(hdr)
        for i, s in enumerate(seeds):
            row = f"{s:<6}"
            for k in ['KL', 'Top1']:
                row += f"{Lm[k][i]:>14.4f}{Fm[k][i]:>14.4f}"
            print(row)
        print()

    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        have_scipy = False

    print(f"=== Multi-seed metrics on the SAME {len(inter)} POIs (mean ± std, n={len(seeds)}) ===")
    print(f"{'metric':<12}{'leaky w1->w1':>20}{'free w1->w2':>20}{'Δ(F-L)':>12}{'t-test p':>12}{'Wilcoxon p':>12}")
    for k in keys:
        lm, ls = mean_std(Lm[k])
        fm, fs = mean_std(Fm[k])
        d = np.asarray(Fm[k]) - np.asarray(Lm[k])
        dm = float(d.mean())
        p_t = p_w = float('nan')
        if have_scipy and len(seeds) >= 2 and np.any(d != 0):
            try:
                p_t = float(stats.ttest_rel(Fm[k], Lm[k]).pvalue)
            except Exception:
                pass
            try:
                p_w = float(stats.wilcoxon(Fm[k], Lm[k]).pvalue)
            except Exception:
                pass
        print(f"{k:<12}{lm:>11.4f}±{ls:<8.4f}{fm:>11.4f}±{fs:<8.4f}{dm:>12.4f}{p_t:>12.4g}{p_w:>12.4g}")
    print()
    print("Δ(F-L) = mean(free - leaky) across seeds (negative = leakage-free is lower/worse).")
    if have_scipy:
        print("t-test p = two-sided paired t-test; Wilcoxon p = signed-rank (nonparametric).")
        print(f"Pairing is by shared training seed. With n={len(seeds)} seeds, p-values are indicative;")
        print("a metric whose Δ is small AND not significant supports 'performance not inflated by leakage'.")
    else:
        print("(scipy unavailable -> p-values skipped; `pip install scipy` to enable significance tests.)")


if __name__ == "__main__":
    main()
