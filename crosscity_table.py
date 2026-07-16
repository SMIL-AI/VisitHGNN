#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosscity_table.py  -  cross-city / cross-setting comparison table (R3#6 / R1#3
/ R3#7 / R1#4).

UNLIKE baseline_table.py (which scores every method on the SAME common POI
intersection), cities have DISJOINT POIs, so each entry is scored on ITS OWN valid
POIs. Use this to compare, e.g.:

    VisitHGNN  Fulton (in-region test)
    VisitHGNN  Athens (independent retrain)        -> R3#6 / R1#3 (another county)
    VisitHGNN  Fulton->Athens (zero-shot transfer) -> R3#7 / R1#4 (unseen region)

Each entry is a `*_preds_test_with_gt.csv` produced by evaluate_and_plot.py
(--save_with_gt or the auto name). Multi-seed entries (a glob of seed files) are
reported as mean±std on their own valid POIs.

Run from the 2026GNN ROOT (so `import eval_on_intersection` resolves):

    python crosscity_table.py \
      --entry "Fulton (test)=/.../MS_w1w2/seed_0/prediction/fulton_w1f_w2l_split_preds_test_with_gt.csv" \
      --entry_glob "Athens (retrain)=/.../athens_*/prediction/athens_split_preds_test_with_gt.csv" \
      --entry "Fulton->Athens (zero-shot)=/.../transfer/athens_split_preds_test_with_gt.csv" \
      --ndcg_k 50 --recall_k 5 \
      --out_csv /.../crosscity_table.csv

Notes
-----
* `--entry "label=path"` is a single CSV; `--entry_glob "label=glob"` is multi-seed
  (mean±std). Order is preserved in the printed table.
* Lower is better for KL/MAE; higher for Top1 / NDCG / Recall.
* n_poi is printed per row because the POI set differs across cities.
"""

import argparse
import glob
import csv
import numpy as np

from eval_on_intersection import load_with_gt, metrics_on, EPS


def _valid_pois(store):
    return sorted({p for p, (pr, gt) in store.items() if gt.sum() > EPS})


def _metrics_for_store(store, ndcg_k, recall_k):
    pois = _valid_pois(store)
    m = metrics_on(pois, store, ndcg_k, recall_k)
    return m, len(pois)


def main():
    ap = argparse.ArgumentParser(description="Cross-city comparison table (each entry on its own valid POIs).")
    ap.add_argument('--entry', action='append', default=[],
                    help='label=path to a single *_with_gt.csv (repeatable)')
    ap.add_argument('--entry_glob', action='append', default=[],
                    help='label=glob for a MULTI-SEED entry (reported as mean±std); repeatable')
    ap.add_argument('--ndcg_k', type=int, default=50)
    ap.add_argument('--recall_k', type=int, default=5)
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    if not args.entry and not args.entry_glob:
        raise SystemExit("[crosscity] pass at least one --entry or --entry_glob")

    keys = ['KL', 'MAE', 'Top1', f'NDCG@{args.ndcg_k}', f'Recall@{args.recall_k}']
    lower_better = {'KL', 'MAE'}
    rows = []  # (label, mean_dict, std_or_None, n_poi)

    # singles
    for spec in args.entry:
        if '=' not in spec:
            raise SystemExit(f"[crosscity] --entry must be label=path, got: {spec}")
        lab, path = spec.split('=', 1)
        store = load_with_gt(path.strip())
        m, n = _metrics_for_store(store, args.ndcg_k, args.recall_k)
        rows.append((lab.strip(), m, None, n))
        print(f"[crosscity] '{lab.strip()}': {n} valid POIs  <- {path.strip()}")

    # multi-seed
    for spec in args.entry_glob:
        if '=' not in spec:
            raise SystemExit(f"[crosscity] --entry_glob must be label=glob, got: {spec}")
        lab, pat = spec.split('=', 1)
        files = sorted(glob.glob(pat.strip()))
        if not files:
            raise SystemExit(f"[crosscity] '{lab.strip()}' matched no files: {pat.strip()}")
        per = []
        ns = []
        for f in files:
            store = load_with_gt(f)
            m, n = _metrics_for_store(store, args.ndcg_k, args.recall_k)
            per.append(m); ns.append(n)
        mean = {k: float(np.mean([m[k] for m in per])) for k in keys}
        std = {k: float(np.std([m[k] for m in per])) for k in keys}
        nmin, nmax = min(ns), max(ns)
        npoi = f"{nmin}" if nmin == nmax else f"{nmin}-{nmax}"
        rows.append((lab.strip(), mean, std if len(files) > 1 else None, npoi))
        print(f"[crosscity] '{lab.strip()}': {len(files)} seed files, "
              f"{npoi} valid POIs each")

    # ---- aligned text table ----
    wlab = max(len(r[0]) for r in rows) + 1
    def cell(m, sd, k):
        return f"{m[k]:.4f}" + (f"±{sd[k]:.4f}" if sd else "")
    hdr = f"{'setting':<{wlab}}" + f"{'n_POI':>9}" + "".join(f"{k:>16}" for k in keys)
    print("\n" + hdr)
    print("-" * len(hdr))
    for lab, m, sd, n in rows:
        print(f"{lab:<{wlab}}" + f"{str(n):>9}" + "".join(f"{cell(m, sd, k):>16}" for k in keys))
    print("\n(lower is better for KL/MAE, higher for the rest; POI sets differ across cities, "
          "so rows are NOT on a shared POI set)")

    # ---- markdown ----
    md = ["", "| Setting | n POI | " + " | ".join(keys) + " |",
          "|" + "---|" * (len(keys) + 2)]
    for lab, m, sd, n in rows:
        md.append("| " + lab + " | " + str(n) + " | " + " | ".join(cell(m, sd, k) for k in keys) + " |")
    print("\n".join(md))

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['setting', 'n_poi'] + keys)
            for lab, m, sd, n in rows:
                wr.writerow([lab, n] + [cell(m, sd, k) for k in keys])
        print(f"\n[crosscity] wrote {args.out_csv}")


if __name__ == '__main__':
    main()
