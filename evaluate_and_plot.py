#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate predictions and visualize with dataset splits (NO 'OTHER' bucket).

Adds POI top-category lookup via:
  node_id/poi_idx -> placekey   (poi_to_cbg_mapping.csv)
  placekey        -> top_category (poi_sample.csv)

If CSVs are not provided, falls back to best-effort extraction from the graph.

Note:
- KL@k has been disabled/removed from outputs and plots.
  The CLI flag --kl_at is kept for compatibility but will be ignored.

Important change:
- We now PREFER matching predictions to ground-truth by (poi_node_id, cbg_node_id)
  to avoid rank-order mismatches caused by equal distances or sorting instability.
  You can control this via --match_by {auto, cbg, rank} (default: auto).
"""

import argparse, os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from contextlib import contextmanager

from train_pp_optimized import to_device_data  # your helper


# ===== US Letter 论文风格（单/双栏宽度与字体）=====
LETTER_ONE_COL = 3.5   # in（单栏典型宽度）
LETTER_TWO_COL = 7.2   # in（双栏典型宽度，<= 8.5in 留边距）

@contextmanager
def paper_rc():
    """仅在需要的总图里套用，不影响 POI 图。"""
    with mpl.rc_context({
        "font.size": 8,         # 正文 8pt
        "axes.titlesize": 9,    # 标题 9pt
        "axes.labelsize": 8,    # 坐标轴标签 8pt
        "xtick.labelsize": 7,   # 刻度 7pt
        "ytick.labelsize": 7,
        "legend.fontsize": 7,   # 图例 7pt
        "figure.dpi": 1200,      # 论文常用 300dpi
        "savefig.dpi": 1200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }):
        yield


# -----------------------------
# Splits
# -----------------------------
def get_splits(data, seed: int = 42):
    def _to_tensor(x):
        if x is None:
            return None
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.long)

    tr = getattr(data, "train_idx", None)
    va = getattr(data, "val_idx", None)
    te = getattr(data, "test_idx", None)

    if tr is None:
        try: tr = data["train_idx"]
        except Exception: pass
    if va is None:
        try: va = data["val_idx"]
        except Exception: pass
    if te is None:
        try: te = data["test_idx"]
        except Exception: pass

    return _to_tensor(tr), _to_tensor(va), _to_tensor(te)


# -----------------------------
# Category resolver from CSVs
# -----------------------------
def build_category_resolver_from_csv(map_csv: str, sample_csv: str):
    """
    Return a function: get_cat(poi_id:int) -> str
    Works fast by prebuilding dicts:
      node_id -> placekey   and   poi_idx -> placekey
      placekey -> top_category (mode or first non-null)
    """
    if not (map_csv and sample_csv and os.path.isfile(map_csv) and os.path.isfile(sample_csv)):
        return None

    # Mapping: node_id/poi_idx -> placekey (keep first per id; strip whitespace)
    head_map = pd.read_csv(map_csv, nrows=0).columns
    use_cols = [c for c in ["node_id", "poi_idx", "placekey"] if c in head_map]
    map_df = pd.read_csv(map_csv, usecols=use_cols)
    if "node_id" in map_df.columns:
        map_df["node_id"] = map_df["node_id"].astype("int64", errors="ignore")
    if "poi_idx" in map_df.columns:
        map_df["poi_idx"] = map_df["poi_idx"].astype("int64", errors="ignore")
    map_df["placekey"] = map_df["placekey"].astype(str).str.strip()

    node2place = map_df.groupby("node_id")["placekey"].first().to_dict() if "node_id" in map_df.columns else {}
    poiidx2place = map_df.groupby("poi_idx")["placekey"].first().to_dict() if "poi_idx" in map_df.columns else {}

    # Sample: placekey -> category (prefer top_category, then macro/sub)
    header = pd.read_csv(sample_csv, nrows=0).columns.tolist()
    use_cols = [c for c in ["placekey", "top_category", "macro_cat", "sub_category"] if c in header]
    sample_df = pd.read_csv(sample_csv, usecols=use_cols)
    # 修复笔误：去除空白
    sample_df["placekey"] = sample_df["placekey"].astype(str).str.strip()

    def _pick_cat(group: pd.DataFrame):
        for col in ("top_category", "macro_cat", "sub_category"):
            if col in group.columns:
                s = group[col].dropna().astype(str)
                if len(s):
                    try:
                        m = s.mode()
                        if len(m):
                            return m.iloc[0]
                    except Exception:
                        pass
                    return s.iloc[0]
        return None

    place2cat = sample_df.groupby("placekey").apply(_pick_cat).to_dict()

    def get_cat(poi_id: int) -> str:
        pk = node2place.get(int(poi_id))
        if pk is None:
            pk = poiidx2place.get(int(poi_id))  # fallback: caller passed poi_idx
        if pk is None:
            return "Unknown"
        cat = place2cat.get(pk)
        return str(cat) if (cat is not None and str(cat).strip() != "") else "Unknown"

    print(f"[CategoryResolver] loaded: {len(node2place)} node_id→placekey, {len(place2cat)} placekey→category.")
    return get_cat


# -----------------------------
# Fallback: try to read from the graph object
# -----------------------------
def resolve_poi_top_category_from_graph(data_cpu, poi_id: int) -> str:
    import numpy as _np, pandas as _pd, torch as _torch
    store = data_cpu['poi']
    # direct text arrays
    for attr in ['top_category', 'category', 'topcat', 'naics_name', 'brand_name']:
        arr = getattr(store, attr, None)
        if isinstance(arr, (list, tuple)) and len(arr) > poi_id:
            return str(arr[poi_id])
        if isinstance(arr, _np.ndarray) and arr.ndim == 1 and len(arr) > poi_id and arr.dtype.kind in ('U','S','O'):
            return str(arr[poi_id])
    # id + name mapping
    for base in ['cat', 'category', 'top_category', 'naics', 'brand']:
        idx = getattr(store, f'{base}_id', None)
        names = getattr(store, f'{base}_names', None)
        if isinstance(idx, _torch.Tensor) and idx.numel() > poi_id and isinstance(names, (list, tuple)):
            j = int(idx[poi_id])
            if 0 <= j < len(names): return str(names[j])
    # embedded DataFrame
    df = getattr(store, 'meta_df', None)
    if isinstance(df, _pd.DataFrame):
        for col in ['top_category', 'category', 'topcat', 'naics_name', 'brand_name']:
            if col in df.columns and poi_id < len(df):
                try: return str(df.iloc[poi_id][col])
                except Exception: pass
    # dict on the bundle
    for key in ['poi_id2top_category','poi_id2category']:
        mp = None
        if isinstance(data_cpu, dict): mp = data_cpu.get(key, None)
        else: mp = getattr(data_cpu, key, None) if hasattr(data_cpu, key) else None
        if isinstance(mp, dict) and poi_id in mp: return str(mp[poi_id])
    return 'Unknown'


# -----------------------------
# Candidates & GT
# -----------------------------
def build_knn_candidates_by_distance(data, K, device):
    e_knn = data[('poi', 'knn', 'cbg')]
    src = e_knn.edge_index[0].cpu().numpy()
    dst = e_knn.edge_index[1].cpu().numpy()
    if getattr(e_knn, 'edge_attr', None) is not None:
        dist = e_knn.edge_attr.view(-1).cpu().numpy()
        df = (pd.DataFrame({'poi': src, 'cbg': dst, 'dist': dist})
              .sort_values(['poi', 'dist']))
    else:
        df = pd.DataFrame({'poi': src, 'cbg': dst, 'dist': 0.0})
    N = int(data['poi'].num_nodes)
    knn = np.full((N, K), -1, np.int64)
    for poi, grp in df.groupby('poi'):
        topk = grp.head(K)['cbg'].astype(int).to_numpy()
        knn[poi, :len(topk)] = topk
    knn_t = torch.tensor(knn, dtype=torch.long, device=device)
    invalid = knn_t.eq(-1)
    return knn_t, invalid


def build_true_probs_from_graph(data, knn, device):
    e_vis = data[('cbg', 'visit', 'poi')]
    cbg = e_vis.edge_index[0].cpu().numpy()
    poi = e_vis.edge_index[1].cpu().numpy()
    w   = e_vis.edge_attr.view(-1).cpu().numpy()
    visit_map = {(int(p), int(c)): float(wt) for p, c, wt in zip(poi, cbg, w)}

    N, K = knn.shape
    true = torch.zeros((N, K), dtype=torch.float32, device=device)
    for i in range(N):
        for j in range(K):
            c = knn[i, j].item()
            if c >= 0:
                true[i, j] = visit_map.get((i, c), 0.0)
    row_sum = true.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    true = true / row_sum
    return true


# -----------------------------
# Metrics (in-candidate only)
# -----------------------------
def metrics_basic(pred, true, invalid, eps=1e-9):
    mask = ~invalid
    p = pred.clone(); p[invalid] = 0; p = p / p.sum(dim=1, keepdim=True).clamp(min=eps)
    t = true.clone(); t[invalid] = 0; t = t / t.sum(dim=1, keepdim=True).clamp(min=eps)
    kl_per = (t * (torch.log(t + eps) - torch.log(p + eps))).sum(1)
    return {
        'KL': kl_per.mean().item(),
        'MAE': (p - t)[mask].abs().mean().item(),
        'Top1': (p.argmax(1)[mask.any(1)] == t.argmax(1)[mask.any(1)]).float().mean().item(),
        'KL_per_poi': kl_per.detach().cpu().numpy(),
    }

def ndcg_at_k(pred, true, invalid, k, eps=1e-9):
    K = pred.size(1); k = max(1, min(int(k), K))
    p = pred.clone(); p[invalid] = 0
    t = true.clone(); t[invalid] = 0
    row_mask = (~invalid).any(1)
    scores = []
    for pi, ti, inv in zip(p[row_mask], t[row_mask], invalid[row_mask]):
        valid = ~inv
        if valid.sum() == 0 or ti[valid].sum() <= 0: continue
        pv, tv = pi[valid], ti[valid]
        order = torch.argsort(pv, descending=True)[:k]
        rel = tv[order]
        idxs = torch.arange(2, 2 + rel.numel(), device=rel.device, dtype=rel.dtype)
        dcg = (rel / torch.log2(idxs)).sum()
        ideal = torch.sort(tv, descending=True).values[:k]
        idcg = (ideal / torch.log2(torch.arange(2, 2 + ideal.numel(), device=ideal.device, dtype=ideal.dtype))).sum().clamp(min=eps)
        scores.append((dcg / idcg).item())
    return float(np.mean(scores)) if scores else float('nan')

def recall_at_k(pred, true, invalid, k, eps=1e-9):

    K = pred.size(1)
    k = max(1, min(int(k), K))

    # 1) 取 Top-k（屏蔽无效位置，确保不会被选中）
    p = pred.clone()
    p[invalid] = -1e9
    topk_idx = p.topk(k, dim=1).indices  # [N, k]

    # 2) 归一化真实分布（仅在有效候选上）
    t = true.clone()
    t[invalid] = 0.0
    row_sum = t.sum(dim=1, keepdim=True)                 # [N,1]
    rows = row_sum.squeeze(1) > 0                        # 只统计真实质量>0的 POI
    # 避免除零：只在 rows 上做归一化
    t[rows] = t[rows] / row_sum[rows]

    # 3) 计算 Top-k 捕获的真实概率质量并求平均
    captured = torch.gather(t, 1, topk_idx).sum(dim=1)   # [N]
    if rows.any():
        return float(captured[rows].mean().item())
    else:
        return float('nan')  # 如果所有行都没有真实质量，可改成 0.0



# -----------------------------
# Pred CSV normalization
# -----------------------------
def normalize_pred_columns(dfp: pd.DataFrame) -> pd.DataFrame:
    df = dfp.copy()
    cols = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n in cols: return cols[n]
        return None
    poi_col  = pick(['poi_node_id','poi','poi_id','poi_idx','place_id','place_idx','node_id'])
    cbg_col  = pick(['cbg_node_id','cbg','cbg_id','cbg_idx','dest_id'])
    rank_col = pick(['rank_in_knn','rank','knn_rank','k_rank','position'])
    prob_col = pick(['pred_prob','prob','prediction','pred','score'])
    if poi_col is None:  raise KeyError("Pred CSV missing POI column.")
    if prob_col is None: raise KeyError("Pred CSV missing prediction-prob column.")
    ren = {poi_col: 'poi_node_id', prob_col: 'pred_prob'}
    if cbg_col is not None:  ren[cbg_col]  = 'cbg_node_id'
    if rank_col is not None: ren[rank_col] = 'rank_in_knn'
    df = df.rename(columns=ren)
    df['poi_node_id'] = df['poi_node_id'].astype(int)
    if 'rank_in_knn' in df.columns: df['rank_in_knn'] = df['rank_in_knn'].astype(int)
    if 'cbg_node_id' in df.columns: df['cbg_node_id'] = df['cbg_node_id'].astype(int)
    df['pred_prob'] = df['pred_prob'].astype(float)
    return df


# -----------------------------
# Attach GT to CSV (now: prefer CBG match)
# -----------------------------
def attach_ground_truth_to_csv(dfp, knn, true, save_path, subset_idx=None, match_by='auto'):
    """
    match_by: 'auto' | 'cbg' | 'rank'
      - auto: if cbg_node_id present → use cbg; else rank
    """
    dfp = normalize_pred_columns(dfp)
    knn_np, true_np = knn.cpu().numpy(), true.cpu().numpy()

    # build GT table
    rows = []
    N, K = knn_np.shape
    poi_filter = set(subset_idx.cpu().tolist()) if subset_idx is not None else None
    for i in range(N):
        if poi_filter is not None and i not in poi_filter: continue
        for j in range(K):
            c = int(knn_np[i, j])
            if c >= 0:
                rows.append({'poi_node_id': i, 'rank_in_knn': j+1,
                             'cbg_node_id': c, 'ground_truth': float(true_np[i, j])})
    df_gt = pd.DataFrame(rows)

    # decide match strategy
    has_cbg = 'cbg_node_id' in dfp.columns
    has_rank = 'rank_in_knn' in dfp.columns
    if match_by == 'auto':
        use_cbg = has_cbg
    elif match_by == 'cbg':
        use_cbg = True
    else:
        use_cbg = False  # rank

    if use_cbg and has_cbg:
        df_out = dfp.merge(df_gt.drop(columns='rank_in_knn'),
                           on=['poi_node_id','cbg_node_id'], how='left')
    elif has_rank:
        df_out = dfp.merge(df_gt, on=['poi_node_id','rank_in_knn'], how='left')
    else:
        # last resort: aggregate per-POI GT sum
        agg = df_gt.groupby('poi_node_id', as_index=False)['ground_truth'].sum().rename(columns={'ground_truth':'gt_sum_over_knn'})
        df_out = dfp.merge(agg, on='poi_node_id', how='left')

    cols = list(df_out.columns)
    if 'ground_truth' in cols and 'pred_prob' in cols:
        cols.insert(cols.index('pred_prob'), cols.pop(cols.index('ground_truth')))
        df_out = df_out[cols]

    df_out.to_csv(save_path, index=False)
    return save_path


# -----------------------------
# Consistency check (rank vs cbg)
# -----------------------------
def check_rank_cbg_consistency(dfp, knn):
    """If CSV has both rank_in_knn and cbg_node_id, report rank↔cbg mismatch rate."""
    if not (('rank_in_knn' in dfp.columns) and ('cbg_node_id' in dfp.columns)):
        return
    N, K = int(knn.size(0)), int(knn.size(1))
    mism, tot = 0, 0
    for r in dfp.itertuples(index=False):
        i = int(getattr(r, 'poi_node_id'))
        j = int(getattr(r, 'rank_in_knn')) - 1
        if not (0 <= i < N and 0 <= j < K): 
            continue
        tot += 1
        cbg_csv = int(getattr(r, 'cbg_node_id'))
        cbg_eval = int(knn[i, j].item())
        if cbg_csv != cbg_eval:
            mism += 1
    if tot > 0:
        print(f"[check] rank↔cbg mismatch rate = {mism}/{tot} = {mism/tot:.2%}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', required=True)
    parser.add_argument('--preds_csv', required=True)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', choices=['train','val','test','all'], default='test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--viz', choices=['worst','best','median','random','topk_worst','topk_best','poi'], default='worst')
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--poi', type=int, default=None)
    parser.add_argument('--save_with_gt', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recall_k', type=int, default=5)
    parser.add_argument('--kl_at', type=int, default=None, help='(ignored) kept for CLI compatibility; KL@k is disabled')
    parser.add_argument('--ndcg_k', type=int, default=None)
    parser.add_argument('--match_by', choices=['auto','cbg','rank'], default='auto',
                        help='auto: prefer cbg if present; cbg: force cbg; rank: force rank.')
    # CSVs for top-category lookup
    parser.add_argument('--poi_map_csv', default=None, help='.../fulton_w1_poi_to_cbg_mapping.csv')
    parser.add_argument('--poi_sample_csv', default=None, help='.../fulton_w1_poi_sample.csv')
    args = parser.parse_args()

    # Explicitly ignore KL@k, but warn if user provided it
    if args.kl_at:
        print(f"ℹ️  --kl_at {args.kl_at} is ignored (KL@k disabled).")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # graph
    data_cpu = torch.load(args.graph_path, weights_only=False)
    data = to_device_data(data_cpu, device)

    # splits
    train_idx, val_idx, test_idx = get_splits(data, seed=args.seed)
    if args.split == 'train': subset_idx = train_idx
    elif args.split == 'val': subset_idx = val_idx
    elif args.split == 'test': subset_idx = test_idx
    else: subset_idx = None
    if subset_idx is None:
        subset_idx = torch.arange(int(data['poi'].num_nodes), device=device)
        print("⚠️  No saved splits found or 'all' chosen. Evaluating on ALL POIs.")

    # candidates & GT
    dfp_raw = pd.read_csv(args.preds_csv)
    dfp = normalize_pred_columns(dfp_raw)

    # If we'll match by RANK (either forced or because no cbg column), we can override K from preds
    infer_by_rank = (args.match_by == 'rank') or (args.match_by == 'auto' and 'cbg_node_id' not in dfp.columns)
    if infer_by_rank:
        try:
            if 'rank_in_knn' in dfp.columns and len(dfp['rank_in_knn']) > 0:
                k_from_preds = int(dfp['rank_in_knn'].max())
                if k_from_preds > 0 and k_from_preds != args.k:
                    print(f"Overriding --k={args.k} with K inferred from preds: {k_from_preds}")
                    args.k = k_from_preds
        except Exception:
            pass
    else:
        # Matching by CBG: DO NOT override K from preds to avoid rank-based drift.
        pass

    knn_all, invalid_all = build_knn_candidates_by_distance(data, args.k, device)
    true_all = build_true_probs_from_graph(data, knn_all, device)

    # optional consistency check if both columns exist
    check_rank_cbg_consistency(dfp, knn_all)

    # save preds+GT for chosen split, honoring match_by
    out_gt = (args.save_with_gt or os.path.splitext(args.preds_csv)[0] + f'_{args.split}_with_gt.csv')
    attach_ground_truth_to_csv(dfp, knn_all, true_all, out_gt, subset_idx=subset_idx, match_by=args.match_by)
    print(f"[Saved] preds with_ground_truth → {out_gt}")

    # build pred tensor (prefer cbg; fallback rank) according to match_by
    N, K = int(data['poi'].num_nodes), knn_all.size(1)
    pred_all = torch.zeros((N, K), device=device)

    has_cbg = 'cbg_node_id' in dfp.columns
    has_rank = 'rank_in_knn' in dfp.columns
    if args.match_by == 'auto':
        use_cbg = has_cbg
    elif args.match_by == 'cbg':
        use_cbg = True
    else:
        use_cbg = False

    if use_cbg and has_cbg:
        # map (poi, cbg) -> column j in evaluator's KNN
        pos = {(i, int(c)): j
               for i in range(N)
               for j, c in enumerate(knn_all[i].tolist()) if c >= 0}
        for r in dfp.itertuples(index=False):
            i = int(getattr(r, 'poi_node_id'))
            c = int(getattr(r, 'cbg_node_id'))
            j = pos.get((i, c))
            if j is not None and 0 <= i < N and 0 <= j < K:
                pred_all[i, j] = float(getattr(r, 'pred_prob'))
    elif has_rank:
        for r in dfp.itertuples(index=False):
            i = int(getattr(r, 'poi_node_id'))
            j = int(getattr(r, 'rank_in_knn')) - 1
            if 0 <= i < N and 0 <= j < K:
                pred_all[i, j] = float(getattr(r, 'pred_prob'))
    else:
        raise KeyError("Pred CSV lacks both 'cbg_node_id' and 'rank_in_knn' for alignment.")

    # subset tensors
    subset_idx = subset_idx.to(device)
    pred = pred_all.index_select(0, subset_idx)
    true = true_all.index_select(0, subset_idx)
    invalid = invalid_all.index_select(0, subset_idx)
    K_local = pred.size(1)

    # metrics (KL@k disabled)
    m_full = metrics_basic(pred, true, invalid)
    ndcg_k_used = max(1, min(args.ndcg_k or K_local, K_local))
    ndcg_k_val = ndcg_at_k(pred, true, invalid, ndcg_k_used)
    recall_k_used = max(1, min(args.recall_k, K_local))
    recall_k_val  = recall_at_k(pred, true, invalid, recall_k_used)
    msg = (f"[Split={args.split}] "
           f"KL={m_full['KL']:.4f}  MAE={m_full['MAE']:.4f}  Top1={m_full['Top1']:.3f}  "
           f"NDCG@{ndcg_k_used}={ndcg_k_val:.3f}  Recall@{recall_k_used}={recall_k_val:.3f}")
    print(msg)

    # category resolver
    cat_resolver = build_category_resolver_from_csv(args.poi_map_csv, args.poi_sample_csv)
    def get_cat(poi_global: int) -> str:
        if cat_resolver is not None:
            return cat_resolver(poi_global)
        return resolve_poi_top_category_from_graph(data_cpu, poi_global)

    # which POIs to plot
    order = np.argsort(m_full['KL_per_poi'])
    rng = np.random.default_rng(args.seed)
    modes = {
        'worst': [int(order[-1])] if len(order) else [],
        'best':  [int(order[0])] if len(order) else [],
        'median':[int(order[len(order)//2])] if len(order) else [],
        'random':[int(rng.integers(len(order)))] if len(order) else [],
        'topk_best': list(order[:args.topk]),
        'topk_worst': list(order[-args.topk:][::-1]),
        'poi': []
    }
    if args.viz == 'poi' and args.poi is not None:
        loc = (subset_idx == int(args.poi)).nonzero(as_tuple=False)
        modes['poi'] = [int(loc.view(-1)[0].item())] if loc.numel() > 0 else modes['worst']
    sel = modes.get(args.viz, modes['worst'])

    # per-POI grouped bars + center lines  —— 不改尺寸与字体（保留原样）
    def plot_one(subset_pos, suffix=""):
        poi_global = int(subset_idx[subset_pos].item())
        centers = np.arange(1, K_local + 1, dtype=float)

        t = true[subset_pos].clone(); t[invalid[subset_pos]] = 0
        p = pred[subset_pos].clone(); p[invalid[subset_pos]] = 0
        t = t / t.sum().clamp(min=1e-9)
        p = p / p.sum().clamp(min=1e-9)
        t_np, p_np = t.detach().cpu().numpy(), p.detach().cpu().numpy()

        step = 1
        if K_local > 80: step = 5
        elif K_local > 40: step = 2

        fig, ax = plt.subplots(figsize=(8, 3))
        c_true, c_pred = 'C0', 'C1'
        bar_w, gap = 0.48, 0.00
        shift = (bar_w/2.0 + gap/2.0)

        ax.bar(centers - shift, t_np, width=bar_w, align='center',
               color=c_true, alpha=0.50, label='True', edgecolor='none')
        ax.bar(centers + shift, p_np, width=bar_w, align='center',
               color=c_pred, alpha=0.50, label='Pred', edgecolor='none')

        ax.set_xlim(centers[0] - 0.6, centers[-1] + 0.6)
        ax.margins(x=0)
        ax.set_xticks(centers[::step])
        ax.set_xticklabels([str(i) for i in range(1, K_local+1, step)], fontsize=8)

        ax.set_xlabel('KNN Neighbor CBGs', fontsize=9)
        ax.set_ylabel('Visit Probability', fontsize=9)

        cat_str = get_cat(poi_global)
        kl_val = float(m_full['KL_per_poi'][subset_pos]) if len(m_full['KL_per_poi']) > subset_pos else float('nan')
        ax.set_title(f"POI {poi_global} | {cat_str} | KL={kl_val:.4f} ", fontsize=10)
        ax.legend(frameon=False)

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f'poi_{poi_global}_bars_{args.split}{suffix}.png')
        fig.savefig(out_path, dpi=600)
        plt.close(fig)
        return out_path

    for idx, subpos in enumerate(sel, 1):
        suffix = f'_{args.viz}_{idx}' if args.viz.startswith('topk') else f'_{args.viz}'
        plot_one(subpos, suffix)

# ============================== FIGURE 1: Overall metrics bar ==============================
    COLORS = {
        "kl":   "#599CB4",
        "top1": "#92B5CA",
        "ndcg": "#AECFD4",
        "r2":   "#CCE4EF",   # reused for Recall
    }
    HATCHES = {
        "kl":   "////",
        "top1": "xx",
        "ndcg": "\\\\\\",
        "r2":   "//",
    }
    MAE_COLOR = "#7A8CA1"  # near-zero MAE color (no hatch)

    metric_names = ['KL', 'MAE', 'Top-1', f'NDCG@{ndcg_k_used}', f'Recall@{recall_k_used}']
    metric_vals  = [m_full['KL'], m_full['MAE'], m_full['Top1'], ndcg_k_val, recall_k_val]
    colors  = [COLORS["kl"], MAE_COLOR, COLORS["top1"], COLORS["ndcg"], COLORS["r2"]]
    hatches = [HATCHES["kl"], None,       HATCHES["top1"], HATCHES["ndcg"], HATCHES["r2"]]

    with paper_rc():
        fig1, ax1 = plt.subplots(figsize=(4, 2.6), layout="constrained")  # 单栏
        bars = ax1.bar(metric_names, metric_vals, width=0.6, color=colors,
                       edgecolor="#2F3B44", linewidth=0.5)
        for b, h in zip(bars, hatches):
            if h: b.set_hatch(h)
        for b, v in zip(bars, metric_vals):
            ax1.text(b.get_x() + b.get_width()/2, b.get_height()+0.01, f"{v:.3f}",
                     ha='center', va='bottom', fontsize=7)
        ax1.set_ylabel('Score')
        ax1.tick_params(direction="in")
        ax1.set_xlabel('')
        ax1.set_ylim(0, max(1.0, max(metric_vals)*1.10))
        fig1.savefig(os.path.join(args.out_dir, f'overall_metrics_{args.split}.png'))
        plt.close(fig1)
# ===========================================================================================

    # ============================== FIGURE 2: KL histogram (4 colors) ==========================
    from matplotlib.patches import Patch
    BIN_COLORS = ["#7F8A9B", "#B7CBD5", "#C1DDDB", "#D1DED7"]  # dark → light

    # Fixed semantic intervals
    EDGES = np.array([0.0, 0.10, 0.25, 0.50, np.inf], dtype=float)
    LABELS = [f"{EDGES[i]:.2f}-{EDGES[i+1]:.2f}" if np.isfinite(EDGES[i+1])
              else r"$\geq$" + f"{EDGES[i]:.2f}" for i in range(4)]

    kl_vals = np.asarray(m_full['KL_per_poi'], dtype=float)

    # Nice x-limit: clip at 99th percentile but not below the 3rd edge
    nice_step = 0.05
    xmax_p = float(np.nanpercentile(kl_vals, 99))
    xmax = max(EDGES[3], np.ceil(xmax_p / nice_step) * nice_step)

    with paper_rc():
        fig2, ax2 = plt.subplots(figsize=(4, 2.6), layout="constrained")  # 双栏
        counts, bin_edges, patches = ax2.hist(kl_vals, bins=40, range=(0.0, xmax),
                                              edgecolor="none", alpha=0.95)

        # Color each bin by its center position
        def color_for_x(xc):
            if xc < EDGES[1]: return BIN_COLORS[0]
            if xc < EDGES[2]: return BIN_COLORS[1]
            if xc < EDGES[3]: return BIN_COLORS[2]
            return BIN_COLORS[3]

        for p, (l, r) in zip(patches, zip(bin_edges[:-1], bin_edges[1:])):
            xc = (l + r) / 2.0
            p.set_facecolor(color_for_x(xc))

        ax2.set_xlabel("KL for each POI")
        ax2.set_ylabel("#POIs")
        ax2.tick_params(direction="in")

        handles = [Patch(facecolor=c, edgecolor="none", label=lab)
                   for c, lab in zip(BIN_COLORS, LABELS)]
        ax2.legend(handles=handles, loc="upper right", frameon=False)

        fig2.savefig(os.path.join(args.out_dir, f'kl_hist_{args.split}_4colors.png'))
        plt.close(fig2)
    # ===========================================================================================

    # ============================== FIGURE 3: Scatter (True vs Pred) ===========================
    mask = (~invalid).detach().cpu().numpy()
    t_in = true.clone(); t_in[invalid] = 0
    p_in = pred.clone(); p_in[invalid] = 0
    rows = (t_in.sum(1) > 0).detach().cpu().numpy()
    if rows.any():
        p_norm = (p_in[rows] / p_in[rows].sum(1, keepdim=True).clamp(min=1e-9)).detach().cpu().numpy()
        t_norm = (t_in[rows] / t_in[rows].sum(1, keepdim=True).clamp(min=1e-9)).detach().cpu().numpy()
        mask_c = mask[rows]
        y_true, y_pred = t_norm[mask_c].ravel(), p_norm[mask_c].ravel()

        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum() + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)

        with paper_rc():
            fig3, ax3 = plt.subplots(figsize=(3.5, 3.5), layout="constrained")  # 单栏正方
            ax3.set_box_aspect(1)

            # Points & 45-degree line (requested colors)
            # ax3.scatter(y_true, y_pred, s=8, alpha=0.35, color="#839DD1", edgecolors='none')
            # ax3.plot([0, 1], [0, 1], linestyle='--', color="#ED6F6E", linewidth=1.2)

            ax3.scatter(y_true, y_pred, s=8, alpha=0.35, color="#96b4d3", edgecolors="#2c4968")
            ax3.plot([0, 1], [0, 1], linestyle='--', color="#B83945", linewidth=1.5)
            ax3.set_xlabel('Observed probability')
            ax3.set_ylabel('Predicted probability')

            # R^2
            ax3.text(0.03, 0.97, rf"$R^2 = {r2:.3f}$",
                     transform=ax3.transAxes, ha='left', va='top', fontsize=8)

            # axis ticks inside for compact look
            ax3.tick_params(direction="in")
            fig3.savefig(os.path.join(args.out_dir, f'scatter_all_{args.split}.png'))
            plt.close(fig3)
    # ===========================================================================================

    print("[Done] All figures saved to:", args.out_dir)


if __name__ == '__main__':
    main()

# python /home/lp43319/projects/GNN/visitgnn/evaluate_and_plot.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --preds_csv /home/lp43319/projects/GNN/visitgnn/output/Train_pp_good/prediction/fulton_w1_hetero_with_text_and_poi_edges_split_preds.csv \
#   --poi_map_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_to_cbg_mapping.csv \
#   --poi_sample_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_sample.csv \
#   --out_dir   /home/lp43319/projects/GNN/visitgnn/output/Train_pp_good/prediction/eva20251001 \
#   --k 50 \
#   --viz topk_best --topk 100 --split test --ndcg_k 50 --recall_k 5 \
#   --match_by auto



# python /home/lp43319/projects/GNN/visitgnn/evaluate_and_plot.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --preds_csv /home/lp43319/projects/GNN/visitgnn/output/prediction1001/fulton_w1_hetero_with_text_and_poi_edges_split_preds.csv \
#   --poi_map_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_to_cbg_mapping.csv \
#   --poi_sample_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_sample.csv \
#   --out_dir   /home/lp43319/projects/GNN/visitgnn/output/prediction/eva2025 \
#   --k 50 \
#   --viz topk_best --topk 100 --split test --ndcg_k 50 --recall_k 5 \
#   --match_by auto



# python /home/lp43319/projects/GNN/visitgnn/evaluate_and_plot.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --preds_csv /home/lp43319/projects/GNN/visitgnn/output/Train_pp_optimized/prediction/fulton_w1_hetero_with_text_and_poi_edges_split_preds.csv \
#   --poi_map_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_to_cbg_mapping.csv \
#   --poi_sample_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_sample.csv \
#   --out_dir   /home/lp43319/projects/GNN/visitgnn/output/Train_pp_optimized/prediction/eva20251001 \
#   --k 50 \
#   --viz topk_best --topk 100 --split test --ndcg_k 50 --recall_k 5 \
#   --match_by auto


# python /home/lp43319/projects/GNN/visitgnn/evaluate_and_plot.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --preds_csv /home/lp43319/projects/GNN/visitgnn/output/Train_pp_good/prediction/fulton_w1_hetero_with_text_and_poi_edges_split_preds.csv \
#   --poi_map_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_to_cbg_mapping.csv \
#   --poi_sample_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_sample.csv \
#   --out_dir   /home/lp43319/projects/GNN/visitgnn/output/Train_pp_good/prediction/eva20251001 \
#   --k 50 \
#   --viz topk_best --topk 150 --split test --ndcg_k 50 --recall_k 5 \
#   --match_by auto
