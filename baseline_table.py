#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_table.py  -  assemble the FINAL baseline comparison table.

Scores VisitHGNN (multi-seed -> mean±std) and every baseline on the SAME common
valid-POI intersection (POIs with a non-zero target in ALL methods), using the
exact evaluate_and_plot metric definitions (reused from eval_on_intersection).
This is the apples-to-apples table for the response-to-reviewers / paper: one
口径, one POI set, every method.

Run from the 2026GNN ROOT (so `import eval_on_intersection` resolves):

    python baseline_table.py \
      --gnn_glob "/home/lp43319/projects/GNN/visitgnn/output/MS_w1w2/seed_*/prediction/*_preds_test_with_gt.csv" \
      --gnn_label VisitHGNN \
      --baseline "GBDT=/home/lp43319/projects/GNN/visitgnn/output/baselines/gbdt/fulton_w1f_w2l_split_gbdt_preds_test_with_gt.csv" \
      --baseline "Gravity=/home/lp43319/projects/GNN/visitgnn/output/baselines/spatial/fulton_w1f_w2l_split_gravity_preds_test_with_gt.csv" \
      --baseline "2SFCA=/home/lp43319/projects/GNN/visitgnn/output/baselines/spatial/fulton_w1f_w2l_split_2sfca_preds_test_with_gt.csv" \
      --baseline "Radiation=/home/lp43319/projects/GNN/visitgnn/output/baselines/spatial/fulton_w1f_w2l_split_radiation_preds_test_with_gt.csv" \
      --ndcg_k 50 --recall_k 5 \
      --out_csv /home/lp43319/projects/GNN/visitgnn/output/baselines/baseline_table.csv

Notes
-----
* The GNN can be a single seed too (just pass one file in the glob); then no std.
* Use `--baseline_glob "RGCN=.../rgcn_seed*_with_gt.csv"` to report a baseline as
  multi-seed mean±std (same treatment as the GNN); `--baseline` stays single-file.
* Lower is better for KL and MAE; higher is better for Top1 / NDCG / Recall.
* `arrow` markers in the printed table show the better direction per column.
"""

import argparse
import glob
import csv
import numpy as np

from eval_on_intersection import load_with_gt, metrics_on, EPS


def _valid_pois(store):
    return {p for p, (pr, gt) in store.items() if gt.sum() > EPS}


def main():
    ap = argparse.ArgumentParser(description="Final baseline comparison table on the common intersection.")
    ap.add_argument('--gnn_glob', required=True, help="glob for GNN seed *_with_gt.csv (1+ files)")
    ap.add_argument('--gnn_label', default='VisitHGNN')
    ap.add_argument('--baseline', action='append', default=[],
                    help='label=path to a baseline *_with_gt.csv (repeatable)')
    ap.add_argument('--baseline_glob', action='append', default=[],
                    help='label=glob for a MULTI-SEED baseline (reported as mean±std, like the GNN); repeatable')
    ap.add_argument('--ndcg_k', type=int, default=50)
    ap.add_argument('--recall_k', type=int, default=5)
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    gnn_files = sorted(glob.glob(args.gnn_glob))
    if not gnn_files:
        raise SystemExit(f"[table] no GNN files matched: {args.gnn_glob}")
    print(f"[table] GNN seed files ({len(gnn_files)}):")
    for f in gnn_files:
        print(f"        {f}")
    gnn_stores = [load_with_gt(f) for f in gnn_files]

    baselines = []
    for spec in args.baseline:
        if '=' not in spec:
            raise SystemExit(f"[table] --baseline must be label=path, got: {spec}")
        lab, path = spec.split('=', 1)
        baselines.append((lab.strip(), load_with_gt(path.strip())))
    print(f"[table] baselines ({len(baselines)}): {', '.join(l for l, _ in baselines)}")

    # multi-seed baselines (reported as mean±std, like the GNN group)
    baseline_globs = []  # (label, files, stores)
    for spec in args.baseline_glob:
        if '=' not in spec:
            raise SystemExit(f"[table] --baseline_glob must be label=glob, got: {spec}")
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat.strip()))
        if not files:
            raise SystemExit(f"[table] --baseline_glob '{lab.strip()}' matched no files: {pat.strip()}")
        baseline_globs.append((lab.strip(), files, [load_with_gt(f) for f in files]))
        print(f"[table] multi-seed baseline '{lab.strip()}' ({len(files)} files):")
        for f in files:
            print(f"        {f}")

    # common valid-POI intersection across GNN seeds AND every baseline (single + multi-seed)
    sets = ([_valid_pois(s) for s in gnn_stores]
            + [_valid_pois(s) for _, _, ss in baseline_globs for s in ss]
            + [_valid_pois(s) for _, s in baselines])
    inter = sorted(set.intersection(*sets)) if sets else []
    if not inter:
        raise SystemExit("[table] empty intersection — check the CSVs share the same POI sampling.")
    print(f"[table] valid POIs per method: "
          f"{[len(s) for s in sets]}  ->  common intersection = {len(inter)}\n")

    keys = ['KL', 'MAE', 'Top1', f'NDCG@{args.ndcg_k}', f'Recall@{args.recall_k}']
    lower_better = {'KL', 'MAE'}

    rows = []  # (label, mean_dict, std_dict_or_None)
    per_seed = [metrics_on(inter, s, args.ndcg_k, args.recall_k) for s in gnn_stores]
    gmean = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
    gstd = {k: float(np.std([m[k] for m in per_seed])) for k in keys}
    rows.append((args.gnn_label, gmean, gstd if len(gnn_files) > 1 else None))
    for lab, files, stores in baseline_globs:
        ps = [metrics_on(inter, s, args.ndcg_k, args.recall_k) for s in stores]
        bmean = {k: float(np.mean([m[k] for m in ps])) for k in keys}
        bstd = {k: float(np.std([m[k] for m in ps])) for k in keys}
        rows.append((lab, bmean, bstd if len(files) > 1 else None))
    for lab, s in baselines:
        rows.append((lab, metrics_on(inter, s, args.ndcg_k, args.recall_k), None))

    # best per metric (for the arrow markers)
    best = {}
    for k in keys:
        vals = [(m[k], lab) for lab, m, _ in rows]
        best[k] = (min if k in lower_better else max)(vals)[1]

    # ---- aligned text table ----
    wlab = max(len(r[0]) for r in rows) + 1
    def fmt(lab, m, sd, k):
        cell = f"{m[k]:.4f}" + (f"±{sd[k]:.4f}" if sd else "")
        return ("*" + cell) if best[k] == lab else (" " + cell)
    hdr = f"{'method':<{wlab}}" + "".join(f"{k:>16}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for lab, m, sd in rows:
        print(f"{lab:<{wlab}}" + "".join(f"{fmt(lab, m, sd, k):>16}" for k in keys))
    print("\n(* = best in column; lower is better for KL/MAE, higher for the rest; "
          f"all on the same {len(inter)} POIs)")

    # ---- markdown (paste into the response/paper) ----
    md = ["", "| Method | " + " | ".join(keys) + " |",
          "|" + "---|" * (len(keys) + 1)]
    for lab, m, sd in rows:
        cells = [(f"**{m[k]:.4f}" + (f"±{sd[k]:.4f}" if sd else "") + "**")
                 if best[k] == lab else (f"{m[k]:.4f}" + (f"±{sd[k]:.4f}" if sd else ""))
                 for k in keys]
        md.append("| " + lab + " | " + " | ".join(cells) + " |")
    print("\n".join(md))

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['method'] + keys)
            for lab, m, sd in rows:
                wr.writerow([lab] + [(f"{m[k]:.4f}±{sd[k]:.4f}" if sd else f"{m[k]:.4f}") for k in keys])
        print(f"\n[table] wrote {args.out_csv}")


if __name__ == '__main__':
    main()
