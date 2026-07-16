#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_hetero_gnn.py  -  RGCN / HAN heterogeneous-GNN baselines (R2#3 / R3#9).

Purpose
-------
Reviewers asked for standard heterogeneous-GNN baselines (HAN, RGCN, CompGCN).
This script provides RGCN and HAN under an IDENTICAL protocol to VisitHGNN, so
that the ONLY thing that differs from the proposed model is the message-passing
backbone. It does this by REUSING VisitHGNN's own building blocks, imported
directly from train_pp_optimized.py:

    - the same graph (the same *_split.pt),
    - the same message-passing edge set: build_edge_index_full + filter_edges_for_use
      with EXACTLY the flags train() uses -- POI-POI{poi_poi_modes}, CBG-CBG{adjacent},
      POI-CBG{belong,knn} + reverse edges; the ('cbg','visit','poi') target edges are
      EXCLUDED (include_visit_edges=False),
    - the same candidate set + masked-KL targets (build_targets_from_knn_candidates),
    - the same train/val/test split (get_splits, seed 42),
    - the same masked-KL loss and metrics (masked_kl_loss / masked_metrics),
    - the SAME pairwise prediction head as VisitHGNN: an MLP on
      [cbg_emb || poi_emb] -> per-POI softmax over the K candidates
      (PairScorer below mirrors VisitHGNN.pred_mlp / predict_logits / predict_probs).

Only the encoder changes:
    - rgcn : per-node-type input projection -> homogeneous graph with integer
             edge types -> stacked RGCNConv -> split back to poi/cbg embeddings.
    - han  : per-node-type input projection -> stacked HANConv on the
             heterogeneous edge_index_dict (each edge type = a one-hop metapath).

Fair-comparison note: standard RGCN and HAN (PyTorch Geometric) do NOT consume
edge attributes. VisitHGNN's edge-attribute modelling (geo/temporal distances,
learnable temperatures, relation gating) is part of the proposed contribution, so
these baselines use the same graph STRUCTURE but not the edge attributes -- the
faithful, standard form of these models. CompGCN is intentionally omitted: it is a
knowledge-graph (typed, directed, entity-relation-composition) model and maps
poorly onto this two-node-type weighted graph; RGCN and HAN are the natural fits.

Distance-augmented variant (--use_dist_feature): because vanilla RGCN/HAN cannot
see the per-candidate distance (the dominant ranking signal, which GBDT gets as an
input column and VisitHGNN injects via weighted message passing), this flag adds
per-candidate [standardised distance, normalised rank] to the SAME prediction head.
It gives the baselines the same distance access as GBDT, so the comparison cannot
be dismissed as "crippled baselines": if RGCN/HAN still trail VisitHGNN the
conclusion is stronger, and if they catch GBDT the gap is localised to distance
access rather than architecture. Outputs are tagged '_dist' so both runs coexist.

Output
------
For each model and seed, a preds CSV in the EXACT inferencecopy format
    columns: poi_node_id, rank_in_knn, cbg_node_id, pred_prob
so the SAME evaluate_and_plot.py scores it identically, and it drops straight into
eval_on_intersection.py / baseline_table.py against VisitHGNN and GBDT.

Run (cluster venv with torch + torch_geometric; run from the 2026GNN repo root so
train_pp_optimized.py is importable):
    python baseline_hetero_gnn.py \
        --graph_path /path/to/<graph>_split.pt \
        --out_dir    /path/to/out/hetero_gnn \
        --models rgcn,han --seeds 0 1 2 3 4 --epochs 1000

Then score each CSV exactly like the model / GBDT:
    python evaluate_and_plot.py \
        --graph_path /path/to/<graph>_split.pt \
        --preds_csv  /path/to/out/hetero_gnn/<graph>_split_rgcn_seed0_preds.csv \
        --split test --ndcg_k 50 --recall_k 5 --match_by auto --save_with_gt
(and likewise for han / each seed), then aggregate with baseline_table.py.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[hetero] PyTorch is required. Run inside the project venv on the cluster.\n"
        f"  import error: {e}"
    )

try:
    from torch_geometric.nn import RGCNConv, HANConv
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[hetero] torch_geometric (with RGCNConv, HANConv) is required.\n"
        f"  import error: {e}"
    )

# --- reuse VisitHGNN's own building blocks so the protocol is identical ----- #
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from train_pp_optimized import (
        to_device_data, build_edge_index_full, filter_edges_for_use,
        build_targets_from_knn_candidates, masked_kl_loss, masked_metrics, get_splits,
    )
    try:
        from train_pp_optimized import set_seed as _set_seed
    except Exception:
        def _set_seed(s):
            import random
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[hetero] Could not import train_pp_optimized.py. Run this script from the\n"
        "2026GNN repo root (where train_pp_optimized.py lives), copy it alongside,\n"
        "or put its folder on PYTHONPATH.\n"
        f"  import error: {e}"
    )


# --------------------------------------------------------------------------- #
#  Shared pairwise scoring head  (identical to VisitHGNN.predict_logits)       #
# --------------------------------------------------------------------------- #
class PairScorer(nn.Module):
    """MLP on [cbg_emb || poi_emb (|| per-candidate distance features)] ->
    per-POI softmax over K candidates. With extra_dim=0 this mirrors
    VisitHGNN.pred_mlp / predict_logits / predict_probs exactly; with extra_dim>0
    it ALSO consumes per-candidate distance features (the --use_dist_feature
    variant), giving the baseline the same per-candidate distance access GBDT has."""
    def __init__(self, d_hidden: int, dropout: float, extra_dim: int = 0):
        super().__init__()
        self.extra_dim = int(extra_dim)
        self.pred_mlp = nn.Sequential(
            nn.Linear(2 * d_hidden + self.extra_dim, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def predict_logits(self, z, knn_idx: torch.Tensor, invalid_mask: torch.Tensor, extra: torch.Tensor = None):
        poi_e, cbg_e = z['poi'], z['cbg']
        N, K = knn_idx.shape
        filled = knn_idx.clone()
        filled[invalid_mask] = 0
        cbg_knn = cbg_e[filled]                          # [N, K, H]
        poi_rep = poi_e.unsqueeze(1).expand(N, K, -1)    # [N, K, H]
        parts = [cbg_knn, poi_rep]
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError("PairScorer has extra_dim>0 but no `extra` distance features were passed.")
            parts.append(extra)                          # [N, K, extra_dim]
        pair = torch.cat(parts, dim=-1)                  # [N, K, 2H(+extra)]
        logits = self.pred_mlp(pair.reshape(N * K, -1)).view(N, K)
        return logits.masked_fill(invalid_mask, float('-inf'))

    def predict_probs(self, z, knn_idx: torch.Tensor, invalid_mask: torch.Tensor, extra: torch.Tensor = None):
        return torch.softmax(self.predict_logits(z, knn_idx, invalid_mask, extra), dim=1).masked_fill(invalid_mask, 0.0)


# --------------------------------------------------------------------------- #
#  Homogeneous conversion for RGCN  (POI nodes first, then CBG nodes)          #
# --------------------------------------------------------------------------- #
def build_homogeneous(edge_index_dict, n_poi, n_cbg, device):
    """Concatenate all relations into a single (edge_index, edge_type) with
    global node ids: poi -> [0, n_poi), cbg -> [n_poi, n_poi+n_cbg)."""
    offset = {'poi': 0, 'cbg': n_poi}
    rel2id, eis, ets = {}, [], []
    for et, ei in edge_index_dict.items():
        s, _, d = et
        if s not in offset or d not in offset:
            raise ValueError(f"Unexpected node type in edge {et}; expected only poi/cbg.")
        rid = rel2id.setdefault(et, len(rel2id))
        gi = ei.clone()
        gi[0] = ei[0] + offset[s]
        gi[1] = ei[1] + offset[d]
        eis.append(gi)
        ets.append(torch.full((ei.size(1),), rid, dtype=torch.long, device=ei.device))
    edge_index = torch.cat(eis, dim=1).to(device)
    edge_type = torch.cat(ets, dim=0).to(device)
    return edge_index, edge_type, len(rel2id), rel2id


def build_dist_features(data, knn_idx, invalid, device):
    """Per-candidate distance features aligned with knn_idx, for --use_dist_feature.
    Distances are looked up PER (poi, cbg) pair directly from the ('poi','knn','cbg')
    edges -- NOT by relying on sort order -- so the feature is guaranteed aligned with
    the candidate set; any candidate without a knn distance aborts the run.
    Returns float tensor [N, K, 2] = [standardised distance, normalised rank], with
    invalid cells zeroed (they are masked out in the head anyway)."""
    et = ('poi', 'knn', 'cbg')
    store = data[et]
    if getattr(store, 'edge_attr', None) is None:
        raise RuntimeError("[hetero] ('poi','knn','cbg') has no edge_attr (distance); "
                           "cannot build --use_dist_feature for this graph.")
    ei = store.edge_index.detach().cpu().numpy()
    ea = store.edge_attr.detach().cpu().numpy().reshape(ei.shape[1], -1)[:, 0]  # col0 = distance (km)
    dmap = {(int(s), int(d)): float(w) for s, d, w in zip(ei[0], ei[1], ea)}

    ki = knn_idx.detach().cpu().numpy()
    val = ~invalid.detach().cpu().numpy()
    N, K = ki.shape
    dist_mat = np.zeros((N, K), dtype=np.float64)
    miss = 0
    for i in range(N):
        row = ki[i]; vr = val[i]
        for j in range(K):
            if not vr[j]:
                continue
            w = dmap.get((i, int(row[j])))
            if w is None:
                miss += 1
            else:
                dist_mat[i, j] = w
    if miss:
        raise RuntimeError(f"[hetero] {miss} candidate (poi,cbg) pairs had no knn distance; "
                           "the candidate set and knn edges are inconsistent -- aborting "
                           "rather than feeding a wrong distance feature.")

    dvals = dist_mat[val]
    mu, sd = float(dvals.mean()), float(dvals.std() + 1e-8)
    dist_z = (dist_mat - mu) / sd
    dist_z[~val] = 0.0
    rank_norm = (np.tile(np.arange(1, K + 1, dtype=np.float64), (N, 1)) / float(K))
    rank_norm[~val] = 0.0
    feats = np.stack([dist_z, rank_norm], axis=-1)  # [N, K, 2]
    print(f"[hetero] --use_dist_feature ON: per-candidate [dist_z, rank_norm] "
          f"(extra_dim=2; knn distance km mean={mu:.3f} std={sd:.3f})")
    return torch.tensor(feats, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
#  Encoders                                                                    #
# --------------------------------------------------------------------------- #
class RGCNEncoder(nn.Module):
    def __init__(self, d_poi_feat, d_cbg_feat, d_hidden, num_relations, layers, dropout, num_bases=None, extra_dim=0):
        super().__init__()
        self.poi_in = nn.Linear(d_poi_feat, d_hidden)
        self.cbg_in = nn.Linear(d_cbg_feat, d_hidden)
        self.convs = nn.ModuleList([
            RGCNConv(d_hidden, d_hidden, num_relations=num_relations, num_bases=num_bases)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(layers)])
        self.dropout = dropout
        self.scorer = PairScorer(d_hidden, dropout, extra_dim)

    def encode(self, x_dict, edge_index, edge_type, n_poi):
        x = torch.cat([F.relu(self.poi_in(x_dict['poi'])),
                       F.relu(self.cbg_in(x_dict['cbg']))], dim=0)
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, edge_index, edge_type)
            x = F.relu(norm(x))
            x = F.dropout(x, self.dropout, self.training) + res
        return {'poi': x[:n_poi], 'cbg': x[n_poi:]}


class HANEncoder(nn.Module):
    def __init__(self, d_poi_feat, d_cbg_feat, d_hidden, metadata, heads, layers, dropout, extra_dim=0):
        super().__init__()
        self.poi_in = nn.Linear(d_poi_feat, d_hidden)
        self.cbg_in = nn.Linear(d_cbg_feat, d_hidden)
        self.convs = nn.ModuleList([
            HANConv(d_hidden, d_hidden, metadata=metadata, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.dropout = dropout
        self.scorer = PairScorer(d_hidden, dropout, extra_dim)

    def encode(self, x_dict, edge_index_dict):
        h = {'poi': F.relu(self.poi_in(x_dict['poi'])),
             'cbg': F.relu(self.cbg_in(x_dict['cbg']))}
        for conv in self.convs:
            out = conv(h, edge_index_dict)
            # HANConv returns None for a node type with no incoming metapath;
            # fall back to the previous representation in that case.
            h = {k: (out[k] if out.get(k, None) is not None else h[k]) for k in h}
            h = {k: F.dropout(F.relu(v), self.dropout, self.training) for k, v in h.items()}
        return h


# --------------------------------------------------------------------------- #
#  Train one (model, seed) + write preds CSV                                   #
# --------------------------------------------------------------------------- #
def write_preds_csv(p_all, knn_idx, out_csv):
    N, K = knn_idx.shape
    poi_ids = np.repeat(np.arange(N), K)
    ranks = np.tile(np.arange(1, K + 1), N)
    cbgs = knn_idx.detach().cpu().numpy().reshape(-1)
    probs = p_all.detach().cpu().numpy().reshape(-1)
    keep = cbgs >= 0
    df = pd.DataFrame({
        'poi_node_id': poi_ids[keep].astype(int),
        'rank_in_knn': ranks[keep].astype(int),
        'cbg_node_id': cbgs[keep].astype(int),
        'pred_prob': probs[keep].astype(float),
    }).sort_values(['poi_node_id', 'rank_in_knn']).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return len(df)


def run_one(model_name, data, hetero_edges, homo, knn_idx, true_probs, invalid,
            splits, args, seed, stem, dist_feats=None):
    _set_seed(seed)
    device = args.device
    tr_idx, va_idx, te_idx = splits
    d_poi_feat = int(data['poi'].x.size(1))
    d_cbg_feat = int(data['cbg'].x.size(1))
    n_poi = int(data['poi'].num_nodes)
    n_cbg = int(data['cbg'].num_nodes)

    use_dist = bool(args.use_dist_feature and dist_feats is not None)
    extra_dim = int(dist_feats.size(-1)) if use_dist else 0
    extra = dist_feats if use_dist else None
    tag = model_name + ('_dist' if use_dist else '')

    if model_name == 'rgcn':
        edge_index, edge_type, num_rel, _ = homo
        model = RGCNEncoder(d_poi_feat, d_cbg_feat, args.d_hidden, num_rel,
                            args.layers, args.dropout, num_bases=args.rgcn_num_bases,
                            extra_dim=extra_dim).to(device)

        def fwd():
            return model.encode(data.x_dict, edge_index, edge_type, n_poi)
    elif model_name == 'han':
        metadata = (['poi', 'cbg'], list(hetero_edges.keys()))
        model = HANEncoder(d_poi_feat, d_cbg_feat, args.d_hidden, metadata,
                           args.han_heads, args.layers, args.dropout,
                           extra_dim=extra_dim).to(device)

        def fwd():
            return model.encode(data.x_dict, hetero_edges)
    else:
        raise ValueError(f"unknown model {model_name}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = float('inf'), None

    head = (f" | num_relations={homo[2]}" if model_name == 'rgcn' else f" | heads={args.han_heads}")
    print(f"\n[{tag}] seed={seed} | n_poi={n_poi} n_cbg={n_cbg} "
          f"| d_poi={d_poi_feat} d_cbg={d_cbg_feat} d_hidden={args.d_hidden} "
          f"| layers={args.layers} dropout={args.dropout} lr={args.lr} wd={args.weight_decay} "
          f"epochs={args.epochs} extra_dim={extra_dim}" + head)

    for ep in range(1, args.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        z = fwd()
        p = model.scorer.predict_probs(z, knn_idx, invalid, extra)
        loss = masked_kl_loss(p.index_select(0, tr_idx),
                              true_probs.index_select(0, tr_idx),
                              invalid.index_select(0, tr_idx))
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            z_e = fwd()
            p_e = model.scorer.predict_probs(z_e, knn_idx, invalid, extra)
            m_va = masked_metrics(p_e.index_select(0, va_idx),
                                  true_probs.index_select(0, va_idx),
                                  invalid.index_select(0, va_idx))
        if m_va['kl'] + 1e-6 < best_val:
            best_val = m_va['kl']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            with torch.no_grad():
                m_tr = masked_metrics(p_e.index_select(0, tr_idx),
                                      true_probs.index_select(0, tr_idx),
                                      invalid.index_select(0, tr_idx))
            print(f"[{tag}] Ep{ep:04d} | KL tr {m_tr['kl']:.5f} va {m_va['kl']:.5f} "
                  f"| MAE va {m_va['mae']:.5f} | T1 va {m_va['top1']:.4f} | best_va_KL {best_val:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        z_f = fwd()
        p_all = model.scorer.predict_probs(z_f, knn_idx, invalid, extra)
        m_te = masked_metrics(p_all.index_select(0, te_idx),
                              true_probs.index_select(0, te_idx),
                              invalid.index_select(0, te_idx))
    print(f"[{tag}] seed={seed} TEST | KL {m_te['kl']:.5f} MAE {m_te['mae']:.5f} "
          f"Top-1 {m_te['top1']:.4f} R2 {m_te['r2']:.4f}")

    out_csv = os.path.join(args.out_dir, f"{stem}_{tag}_seed{seed}_preds.csv")
    nrows = write_preds_csv(p_all, knn_idx, out_csv)
    print(f"[{tag}] seed={seed} wrote {out_csv}  ({nrows} rows)")
    return m_te


def main():
    ap = argparse.ArgumentParser(description="RGCN / HAN heterogeneous-GNN baselines (same protocol as VisitHGNN).")
    ap.add_argument('--graph_path', required=True, help="the *_split.pt you train VisitHGNN on")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--models', default='rgcn,han', help="comma list: rgcn,han")
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--k', type=int, default=50, help="candidates per POI (match the model's K)")
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=8.5e-4)
    ap.add_argument('--weight_decay', type=float, default=3e-5)
    ap.add_argument('--d_hidden', type=int, default=64, help="embedding dim (matches VisitHGNN d_hidden)")
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--han_heads', type=int, default=8)
    ap.add_argument('--rgcn_num_bases', type=int, default=None, help="RGCN basis decomposition (default: full)")
    ap.add_argument('--poi_poi_modes', default='geo_knn,time_sim',
                    help="POI-POI relations used as message-passing edges (match your clean run)")
    ap.add_argument('--use_dist_feature', action='store_true',
                    help="feed per-candidate [standardised distance, normalised rank] into the "
                         "prediction head (gives RGCN/HAN the same distance access GBDT/VisitHGNN "
                         "have); outputs get a '_dist' suffix so they don't collide with the vanilla run")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    poi_poi_modes = tuple(m.strip() for m in args.poi_poi_modes.split(',') if m.strip())

    print(f"[hetero] loading graph: {args.graph_path}")
    data = to_device_data(torch.load(args.graph_path, weights_only=False, map_location=args.device), args.device)

    # SAME message-passing edges as VisitHGNN's train() (visit edges EXCLUDED)
    eidx, eattr = build_edge_index_full(data)
    eidx, eattr = filter_edges_for_use(
        eidx, eattr,
        use_poi_poi=True, poi_poi_modes=poi_poi_modes,
        keep_cbg_adj=True, include_visit_edges=False,
        cross_modes=('belong', 'knn'), use_rev_edges=True,
    )
    print(f"[hetero] message-passing edge types used ({len(eidx)}):")
    for et in eidx:
        print(f"          {et}: {eidx[et].size(1)} edges")

    # SAME candidate set + masked-KL targets + split as VisitHGNN
    knn_idx, true_probs, invalid = build_targets_from_knn_candidates(data, args.k, args.device)
    tr_idx, va_idx, te_idx = get_splits(data, 42)
    if te_idx is None:
        raise SystemExit("[hetero] no test split found in the graph; expected data['test_idx'].")
    tr_idx, va_idx, te_idx = tr_idx.to(args.device), va_idx.to(args.device), te_idx.to(args.device)
    splits = (tr_idx, va_idx, te_idx)
    print(f"[hetero] split: train={tr_idx.numel()} val={va_idx.numel()} test={te_idx.numel()} | K={args.k}")

    n_poi = int(data['poi'].num_nodes)
    n_cbg = int(data['cbg'].num_nodes)
    homo = build_homogeneous(eidx, n_poi, n_cbg, args.device) if 'rgcn' in models else None

    # per-candidate distance features (only if --use_dist_feature)
    dist_feats = build_dist_features(data, knn_idx, invalid, args.device) if args.use_dist_feature else None
    suffix = '_dist' if args.use_dist_feature else ''

    stem = os.path.splitext(os.path.basename(args.graph_path))[0]
    summary = {m + suffix: [] for m in models}
    for m in models:
        for s in args.seeds:
            mt = run_one(m, data, eidx, homo, knn_idx, true_probs, invalid, splits, args, s, stem,
                         dist_feats=dist_feats)
            summary[m + suffix].append(mt)

    # tidy mean +/- std over seeds (native test metrics; the 1,060-POI intersection
    # table is produced separately via evaluate_and_plot + baseline_table.py)
    print("\n==================== native test summary (mean +/- std over seeds) ====================")
    for m in models:
        tag = m + suffix
        arr = {k: np.array([d[k] for d in summary[tag]], dtype=float) for k in ('kl', 'mae', 'top1', 'r2')}
        print(f"  {tag:10s} | KL {arr['kl'].mean():.4f}+/-{arr['kl'].std():.4f} "
              f"| MAE {arr['mae'].mean():.4f}+/-{arr['mae'].std():.4f} "
              f"| Top-1 {arr['top1'].mean():.4f}+/-{arr['top1'].std():.4f} "
              f"| R2 {arr['r2'].mean():.4f}+/-{arr['r2'].std():.4f}  (n={len(summary[tag])})")
    print("\n[hetero] next: score each *_preds.csv with evaluate_and_plot.py (--save_with_gt), then")
    print("[hetero] feed the with_gt CSVs to baseline_table.py as extra --baseline entries")
    print("[hetero] (use the SAME 1,060-POI intersection as the VisitHGNN/GBDT table).")


if __name__ == '__main__':
    main()
