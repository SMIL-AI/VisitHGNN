#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ksensitivity_table.py  -  aggregate the multi-seed K-sweep (R3#8 + R2#4).

Reads <root>/K_<K>/seed_<s>/prediction/<stem>_preds_test_with_gt.csv for every K
and seed, and reports, per K (mean ± std over seeds, on the common valid-POI
intersection), THREE families of metrics:

  1. native-@K   : KL / MAE / Top1 / NDCG@50 / Recall@5 computed over each K's own
                   candidate set (the gt is renormalised over K). These describe
                   the model AT that operating point but are NOT directly
                   comparable across K (more candidates spread probability), so
                   they are reported next to coverage.
  2. coverage(K) : fraction of the FULL in-region visit mass captured by the K
                   nearest candidates (model-independent; from the graph). Tells
                   you how much real signal each K's candidate set even contains.
  3. vs-FULL     : Top1 and Recall@k measured against the FULL 544-CBG target
                   distribution (a FIXED reference), so they ARE comparable across
                   K and directly answer "does enlarging K recover real origins
                   the candidate set was missing?". Requires --graph_path.

Run from the 2026GNN ROOT (so `import eval_on_intersection` resolves):
    python ksensitivity_table.py \
        --root       /home/lp43319/projects/GNN/visitgnn/output/KS_w1w2 \
        --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_split.pt \
        --stem       fulton_w1f_w2l_split \
        --ndcg_k 50 --recall_k 5 \
        --out_csv    /home/lp43319/projects/GNN/visitgnn/output/KS_w1w2/ksensitivity_table.csv
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

from eval_on_intersection import load_with_gt, metrics_on, EPS


# --------------------------------------------------------------------------- #
#  torch-free cores                                                           #
# --------------------------------------------------------------------------- #
def load_pred_by_cbg(path):
    """Return {poi: {cbg: pred_prob}} for test POIs (rows with non-NaN gt)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    poi_c = pick(['poi_node_id', 'poi', 'poi_id', 'poi_idx', 'node_id'])
    cbg_c = pick(['cbg_node_id', 'cbg', 'cbg_id', 'cbg_idx'])
    prob_c = pick(['pred_prob', 'pred_prob_x', 'pred_prob_y', 'prob', 'prediction', 'pred', 'score'])
    gt_c = pick(['ground_truth', 'ground_truth_y', 'ground_truth_x', 'gt', 'true', 'target'])
    if poi_c is None or cbg_c is None or prob_c is None:
        raise KeyError(f"{path}: need poi/cbg/pred columns; got {list(df.columns)}")
    if gt_c is not None:
        df = df[~df[gt_c].isna()]
    out = {}
    for poi, g in df.groupby(poi_c):
        out[int(poi)] = dict(zip(g[cbg_c].astype(int), g[prob_c].astype(float)))
    return out


def fulltarget_metrics(pois, t_full, pred_by_cbg, recall_k):
    """Top1 and Recall@k of the model's predictions against the FULL 544-CBG
    target t_full[poi] = {cbg: weight}. Comparable across K."""
    top1, rec, n = 0, [], 0
    for poi in pois:
        t = t_full.get(poi)
        pr = pred_by_cbg.get(poi)
        if not t or not pr:
            continue
        tsum = sum(t.values())
        if tsum <= EPS:
            continue
        n += 1
        true_top = max(t, key=t.get)                       # true #1 origin CBG
        pred_top = max(pr, key=pr.get)                     # model's #1 (over its K candidates)
        if pred_top == true_top:
            top1 += 1
        topk = sorted(pr, key=pr.get, reverse=True)[:recall_k]
        rec.append(sum(t.get(c, 0.0) for c in topk) / tsum)  # true mass captured by pred top-k
    return {
        'n': n,
        'Top1_full': (top1 / n) if n else float('nan'),
        f'Recall@{recall_k}_full': float(np.mean(rec)) if rec else float('nan'),
    }


def coverage_at_K(pois, t_full, knn_rank, K):
    """Mean over pois of (full-target mass within the K nearest candidates)."""
    covs = []
    for poi in pois:
        t = t_full.get(poi)
        order = knn_rank.get(poi)
        if not t or not order:
            continue
        tsum = sum(t.values())
        if tsum <= EPS:
            continue
        topK = set(order[:K])
        covs.append(sum(w for c, w in t.items() if c in topK) / tsum)
    return float(np.mean(covs)) if covs else float('nan')


# --------------------------------------------------------------------------- #
def _load_graph_full(graph_path):
    """Return t_full[poi]={cbg:w_norm} (full in-region target) and
    knn_rank[poi]=[cbg ... nearest-first] from the graph."""
    import torch
    data = torch.load(graph_path, weights_only=False)
    ev = data[('cbg', 'visit', 'poi')]
    vc = ev.edge_index[0].cpu().numpy()
    vp = ev.edge_index[1].cpu().numpy()
    vw = ev.edge_attr.view(-1).cpu().numpy()
    t_full = {}
    for c, p, w in zip(vc, vp, vw):
        t_full.setdefault(int(p), {})[int(c)] = float(w)

    ek = data[('poi', 'knn', 'cbg')]
    es = ek.edge_index[0].cpu().numpy()
    ed = ek.edge_index[1].cpu().numpy()
    edist = (ek.edge_attr.view(-1).cpu().numpy()
             if getattr(ek, 'edge_attr', None) is not None else np.zeros_like(es, float))
    kdf = pd.DataFrame({'poi': es, 'cbg': ed, 'd': edist}).sort_values(['poi', 'd'])
    knn_rank = {int(p): g['cbg'].astype(int).tolist() for p, g in kdf.groupby('poi')}
    return t_full, knn_rank


def _parse_K_seed(path):
    mK = re.search(r'[/_]K[_=]?(\d+)', path)
    mS = re.search(r'seed[_=]?(\d+)', path)
    return (int(mK.group(1)) if mK else None, int(mS.group(1)) if mS else None)


def _ms(xs):
    a = np.asarray(xs, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description="Aggregate the multi-seed K-sweep (R3#8 + R2#4).")
    ap.add_argument('--root', required=True, help="KS base dir containing K_<K>/seed_<s>/")
    ap.add_argument('--stem', required=True, help="graph stem, e.g. fulton_w1f_w2l_split")
    ap.add_argument('--graph_path', default=None, help="enables coverage(K) + vs-FULL metrics")
    ap.add_argument('--ndcg_k', type=int, default=50)
    ap.add_argument('--recall_k', type=int, default=5)
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    pattern = os.path.join(args.root, 'K_*', 'seed_*', 'prediction', f'{args.stem}_preds_test_with_gt.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[ks] no files matched: {pattern}")
    byK = {}
    for f in files:
        K, s = _parse_K_seed(f)
        if K is None:
            print(f"[ks] WARN: could not parse K from {f}; skipping"); continue
        byK.setdefault(K, []).append((s, f))
    Ks = sorted(byK)
    print(f"[ks] found {len(files)} runs across K = {Ks}")
    for K in Ks:
        print(f"     K={K}: seeds {sorted(s for s, _ in byK[K])}")

    # stores
    native_stores = {K: {s: load_with_gt(f) for s, f in byK[K]} for K in Ks}
    # common valid-POI intersection across EVERY run
    def valids(store):
        return {p for p, (pr, gt) in store.items() if gt.sum() > EPS}
    all_sets = [valids(native_stores[K][s]) for K in Ks for s, _ in byK[K]]
    inter = sorted(set.intersection(*all_sets))
    print(f"[ks] common valid-POI intersection across all runs: {len(inter)} POIs\n")

    full = None
    if args.graph_path:
        print(f"[ks] loading graph for coverage + vs-FULL metrics: {args.graph_path}")
        t_full, knn_rank = _load_graph_full(args.graph_path)
        pred_stores = {K: {s: load_pred_by_cbg(f) for s, f in byK[K]} for K in Ks}
        full = (t_full, knn_rank, pred_stores)

    nkeys = ['KL', 'MAE', 'Top1', f'NDCG@{args.ndcg_k}', f'Recall@{args.recall_k}']
    rows = []
    for K in Ks:
        row = {'K': K, 'n_seeds': len(byK[K])}
        # native-@K
        per = [metrics_on(inter, native_stores[K][s], args.ndcg_k, args.recall_k) for s, _ in byK[K]]
        for k in nkeys:
            m, sd = _ms([p[k] for p in per])
            row[k] = m; row[k + '_std'] = sd
        # coverage + vs-FULL
        if full:
            t_full, knn_rank, pred_stores = full
            row['coverage'] = coverage_at_K(inter, t_full, knn_rank, K)
            ft = [fulltarget_metrics(inter, t_full, pred_stores[K][s], args.recall_k) for s, _ in byK[K]]
            m, sd = _ms([p['Top1_full'] for p in ft]); row['Top1_full'] = m; row['Top1_full_std'] = sd
            rk = f'Recall@{args.recall_k}_full'
            m, sd = _ms([p[rk] for p in ft]); row[rk] = m; row[rk + '_std'] = sd
        rows.append(row)

    # ---- print ----
    def cell(row, k):
        return f"{row[k]:.4f}±{row[k+'_std']:.4f}"
    print("=== native-@K (each on its own K-candidate set; report WITH coverage) ===")
    hdr = f"{'K':>5} {'cover':>8} | " + " ".join(f"{k:>16}" for k in nkeys)
    print(hdr); print('-' * len(hdr))
    for row in rows:
        cov = f"{row.get('coverage', float('nan')):.3f}" if 'coverage' in row else '   -  '
        print(f"{row['K']:>5} {cov:>8} | " + " ".join(f"{cell(row,k):>16}" for k in nkeys))

    if full:
        print("\n=== vs-FULL 544-CBG target (FIXED reference -> comparable across K) ===")
        rk = f'Recall@{args.recall_k}_full'
        h2 = f"{'K':>5} {'cover':>8} {'Top1_full':>16} {rk:>18}"
        print(h2); print('-' * len(h2))
        for row in rows:
            print(f"{row['K']:>5} {row['coverage']:>8.3f} {cell(row,'Top1_full'):>16} {cell(row,rk):>18}")
        print("\nRead: coverage and the vs-FULL metrics should RISE with K as the candidate set "
              "captures more true origins. If native-@K Top1/NDCG hold up while these rise, a larger "
              "K is strictly better; if they plateau, the knee is your operating point.")

    # ---- markdown ----
    md_keys = (['coverage'] if full else []) + nkeys + ([f'Top1_full', f'Recall@{args.recall_k}_full'] if full else [])
    md = ["", "| K | " + " | ".join(md_keys) + " |", "|" + "---|" * (len(md_keys) + 1)]
    for row in rows:
        cells = []
        for k in md_keys:
            cells.append(f"{row[k]:.3f}" if k == 'coverage' else cell(row, k))
        md.append(f"| {row['K']} | " + " | ".join(cells) + " |")
    print("\n".join(md))

    if args.out_csv:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\n[ks] wrote {args.out_csv}")


if __name__ == '__main__':
    main()
