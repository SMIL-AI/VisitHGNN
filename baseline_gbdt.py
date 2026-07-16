#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_gbdt.py  -  "same features, NO message passing" tabular baseline.

Purpose (addresses R3#9 / R2#3 / R1: "baselines are too weak"):
    A gradient-boosted-tree regressor (sklearn HistGradientBoostingRegressor)
    that sees EXACTLY the same per-node features and the SAME candidate set as
    VisitHGNN, but has no access to the graph structure / message passing.
    For every (POI, candidate-CBG) pair it regresses the normalised visit
    probability from:
        [ POI node features (Dp) | CBG node features (Dc) | distance_km | rank ]
    It is trained on the train-split POIs, predicts for every POI, then
    re-normalises the predictions per POI into a distribution over that POI's
    candidates.

    Because it consumes the same features and candidates, the gap between this
    baseline and VisitHGNN isolates the value added by the heterogeneous graph
    structure itself.

Output:
    A preds CSV in the EXACT inferencecopy format
        columns: poi_node_id, rank_in_knn, cbg_node_id, pred_prob
    so the SAME evaluate_and_plot.py (and eval_on_intersection.py /
    aggregate_seeds.py) score it identically to the model.

This script is SELF-CONTAINED: it reads only the graph .pt (node features,
('poi','knn','cbg') candidate edges + km distances, ('cbg','visit','poi') visit
weights, and the top-level train/val/test_idx). No external CSV is required.

Run (on the cluster, inside the venv with torch):
    python baseline_gbdt.py \
        --graph_path /path/to/<graph>_split.pt \
        --out_dir    /path/to/out/gbdt
Then score it with the SAME flags you use for the model:
    python evaluate_and_plot.py \
        --graph_path /path/to/<graph>_split.pt \
        --preds_csv  /path/to/out/gbdt/<graph>_split_gbdt_preds.csv \
        --split test --ndcg_k 50 --recall_k 5 --match_by auto
(adjust --preds_csv flag name to your evaluate_and_plot; see its --help)
"""

import argparse
import os
import numpy as np
import pandas as pd

try:
    import torch
except Exception as e:  # pragma: no cover - torch always present on the cluster
    raise SystemExit(
        "[gbdt] PyTorch is required to load the graph .pt. Run inside the "
        "project venv on the cluster (login node is fine; no CUDA needed).\n"
        f"  import error: {e}"
    )

from sklearn.ensemble import HistGradientBoostingRegressor


# --------------------------------------------------------------------------- #
#  Candidate / target extraction  (mirrors train_pp_optimized.build_targets,  #
#  additionally returning the per-candidate distance aligned to the ranking)  #
# --------------------------------------------------------------------------- #
def extract_candidates(data, K):
    """Return knn_idx[N,K] (cbg node ids, -1 pad, nearest-first),
    dist_km[N,K] (km, aligned to knn_idx), true_p[N,K] (per-POI normalised).

    Candidates are sorted by ['poi','dist'] exactly like the trainer, so column
    j corresponds to rank_in_knn = j+1 (1 = nearest)."""
    e_knn = data[('poi', 'knn', 'cbg')]
    src = e_knn.edge_index[0].cpu().numpy()
    dst = e_knn.edge_index[1].cpu().numpy()
    if getattr(e_knn, 'edge_attr', None) is not None:
        dkm = e_knn.edge_attr.view(-1).cpu().numpy().astype(float)
    else:
        dkm = np.zeros_like(src, dtype=float)
    df = pd.DataFrame({'poi': src, 'cbg': dst, 'dist': dkm}).sort_values(['poi', 'dist'])

    e_vis = data[('cbg', 'visit', 'poi')]
    v_cbg = e_vis.edge_index[0].cpu().numpy()
    v_poi = e_vis.edge_index[1].cpu().numpy()
    v_w = e_vis.edge_attr.view(-1).cpu().numpy()
    visit_map = {(int(p), int(c)): float(w) for c, p, w in zip(v_cbg, v_poi, v_w)}

    N = int(data['poi'].num_nodes)
    knn = np.full((N, K), -1, dtype=np.int64)
    dist = np.zeros((N, K), dtype=np.float32)
    true_p = np.zeros((N, K), dtype=np.float64)
    for poi, g in df.groupby('poi'):
        tk = g.head(K)
        cbgs = tk['cbg'].to_numpy()
        ds = tk['dist'].to_numpy()
        n = len(cbgs)
        knn[poi, :n] = cbgs
        dist[poi, :n] = ds
        ws = np.array([visit_map.get((int(poi), int(c)), 0.0) for c in cbgs], dtype=float)
        s = ws.sum()
        if s > 0:
            true_p[poi, :n] = ws / (s + 1e-8)
    return knn, dist, true_p.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Split fetch  (replicates train_pp_optimized._safe_fetch_split, with a       #
#  seed-42 recompute fallback identical to data_split.py)                      #
# --------------------------------------------------------------------------- #
def _safe_fetch_split(data, key):
    t = getattr(data, key, None)
    if isinstance(t, torch.Tensor):
        return t
    try:
        t = data[key]
        if isinstance(t, torch.Tensor):
            return t
    except Exception:
        pass
    if isinstance(data, dict):
        t = data.get(key, None)
        if isinstance(t, torch.Tensor):
            return t
    return None


def get_split_idx(data, N, seed=42):
    tr = _safe_fetch_split(data, 'train_idx')
    va = _safe_fetch_split(data, 'val_idx')
    te = _safe_fetch_split(data, 'test_idx')
    if tr is not None and te is not None:
        tr = tr.long().view(-1).cpu().numpy()
        va = va.long().view(-1).cpu().numpy() if va is not None else np.array([], dtype=int)
        te = te.long().view(-1).cpu().numpy()
        print(f"[gbdt] using stored split: train={len(tr)} val={len(va)} test={len(te)}")
        return tr, va, te
    # fallback: recompute exactly like data_split.py (seed 42, 0.7/0.15/0.15)
    order = np.random.default_rng(seed).permutation(N)
    n_tr = int(0.7 * N)
    n_va = int(0.15 * N)
    tr, va, te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    print(f"[gbdt] WARNING: stored split not found; recomputed with seed={seed} "
          f"(train={len(tr)} val={len(va)} test={len(te)}). "
          f"Verify this matches your data_split run.")
    return tr, va, te


# --------------------------------------------------------------------------- #
#  Vectorised per-(POI, candidate) row construction                           #
# --------------------------------------------------------------------------- #
def build_rows(knn, dist, true_p, poi_x, cbg_x):
    """Flatten valid (POI, candidate) pairs into a feature matrix, fully
    vectorised (no python-level per-pair concatenation)."""
    N, K = knn.shape
    valid = knn >= 0                                   # [N, K]
    flat = valid.ravel()
    poi_id = np.repeat(np.arange(N), K)[flat]          # POI node id per row
    rank0 = np.tile(np.arange(K), N)[flat]             # 0-based rank per row
    cbg_id = knn.ravel()[flat]                         # CBG node id per row
    d_row = dist.ravel()[flat].astype(np.float32)      # distance km per row
    y_row = true_p.ravel()[flat].astype(np.float32)    # target prob per row

    feats = np.concatenate(
        [
            poi_x[poi_id],                             # [n, Dp]
            cbg_x[cbg_id],                             # [n, Dc]
            d_row[:, None],                            # [n, 1]
            rank0[:, None].astype(np.float32),         # [n, 1]
        ],
        axis=1,
    ).astype(np.float32)
    return poi_id, rank0, cbg_id, feats, y_row


def main():
    ap = argparse.ArgumentParser(description="GBDT tabular baseline on the GNN's features+candidates.")
    ap.add_argument('--graph_path', required=True, help="graph .pt WITH splits (the *_split.pt you train on)")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k', type=int, default=50, help="candidates per POI (match the model's K)")
    ap.add_argument('--max_iter', type=int, default=400)
    ap.add_argument('--learning_rate', type=float, default=0.05)
    ap.add_argument('--max_leaf_nodes', type=int, default=63)
    ap.add_argument('--min_samples_leaf', type=int, default=50)
    ap.add_argument('--l2_regularization', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[gbdt] loading graph: {args.graph_path}")
    data = torch.load(args.graph_path, weights_only=False)
    N = int(data['poi'].num_nodes)

    knn, dist, true_p = extract_candidates(data, args.k)
    poi_x = data['poi'].x.detach().cpu().numpy().astype(np.float32)   # [N, Dp]
    cbg_x = data['cbg'].x.detach().cpu().numpy().astype(np.float32)   # [Ncbg, Dc]
    tr_idx, va_idx, te_idx = get_split_idx(data, N, args.seed)

    poi_id, rank0, cbg_id, feats, y = build_rows(knn, dist, true_p, poi_x, cbg_x)
    print(f"[gbdt] {feats.shape[0]} (POI,candidate) rows; "
          f"feature dim = POI({poi_x.shape[1]}) + CBG({cbg_x.shape[1]}) + dist + rank "
          f"= {feats.shape[1]}")

    # train only on rows whose POI is in the train split
    train_rows = np.isin(poi_id, tr_idx)
    print(f"[gbdt] training on {int(train_rows.sum())} rows "
          f"({len(tr_idx)} train POIs); holding 10% internally for early stopping")

    reg = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=args.seed,
    )
    reg.fit(feats[train_rows], y[train_rows])
    try:
        print(f"[gbdt] fitted {reg.n_iter_} boosting iterations")
    except Exception:
        pass

    pred = np.clip(reg.predict(feats), 0.0, None)

    out = pd.DataFrame({
        'poi_node_id': poi_id,
        'rank_in_knn': rank0 + 1,
        'cbg_node_id': cbg_id,
        '_raw': pred,
    })
    # per-POI normalisation -> proper distribution over each POI's candidates
    denom = out.groupby('poi_node_id')['_raw'].transform('sum')
    out['pred_prob'] = out['_raw'] / (denom + 1e-12)
    out = (out.drop(columns='_raw')
              .sort_values(['poi_node_id', 'rank_in_knn'])
              .reset_index(drop=True))

    stem = os.path.splitext(os.path.basename(args.graph_path))[0]
    out_csv = os.path.join(args.out_dir, f"{stem}_gbdt_preds.csv")
    out.to_csv(out_csv, index=False)

    # quick sanity on the test split (matches evaluate_and_plot's per-POI KL definition)
    te_set = set(int(x) for x in te_idx)
    msk = out['poi_node_id'].isin(te_set).to_numpy()
    print(f"[gbdt] wrote {out_csv}")
    print(f"[gbdt]   total rows={len(out)}  test-POI rows={int(msk.sum())} "
          f"({out.loc[msk,'poi_node_id'].nunique()} test POIs)")
    print(f"[gbdt] next: score with evaluate_and_plot.py --preds_csv {out_csv} "
          f"(same --split/--ndcg_k/--recall_k flags as the model); "
          f"or drop into eval_on_intersection.py against the model's with_gt CSV.")


if __name__ == '__main__':
    main()
