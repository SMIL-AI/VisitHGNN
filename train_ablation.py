#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, json, random, glob
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATv2Conv, HeteroConv, GraphNorm, Linear


# =========================
# Utils & Repro
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_device_data(obj, device):
    if isinstance(obj, dict) and "graph_data" in obj:
        obj = obj["graph_data"]
    assert isinstance(obj, HeteroData), "Expect HeteroData or {'graph_data': HeteroData}"
    return obj.to(device)

def _safe_dir(path: str):
    os.makedirs(path, exist_ok=True); return path

def _dirname(path: str) -> str:
    return os.path.dirname(os.path.abspath(path))

def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# =========================
# Feature utilities
# =========================
def _auto_find_json(dirpath: str, kind: str) -> Optional[str]:
    patt = os.path.join(dirpath, f"*_{kind}_feature_names.json")
    cands = sorted(glob.glob(patt))
    return cands[-1] if cands else None

def _load_feature_names(json_path: Optional[str]) -> Optional[List[str]]:
    if not json_path or (not os.path.isfile(json_path)): return None
    with open(json_path, "r") as f:
        obj = json.load(f)
    cols = obj.get("columns", None)
    if isinstance(cols, list): return [str(c) for c in cols]
    return None

def _build_poi_masks_from_names(names: Optional[List[str]]):
    if not names: return None, None
    text_idx = [i for i, c in enumerate(names) if str(c).startswith("text_")]
    dwell_idx = [i for i, c in enumerate(names) if str(c).startswith("dwell_")]
    return text_idx, dwell_idx

def _apply_feature_variant_inplace(data: HeteroData,
                                   poi_names: Optional[List[str]],
                                   variant: str = "base+text+dwell"):
    """
    variant ∈ {"base", "base+text", "base+dwell", "base+text+dwell"}.
    实现方式：将选择的列置零，不改变维度。
    """
    x = data["poi"].x
    if poi_names is None:
        return
    text_idx, dwell_idx = _build_poi_masks_from_names(poi_names)
    text_idx = torch.tensor(text_idx, dtype=torch.long, device=x.device) if text_idx else None
    dwell_idx = torch.tensor(dwell_idx, dtype=torch.long, device=x.device) if dwell_idx else None
    with torch.no_grad():
        if variant == "base":
            if text_idx is not None and text_idx.numel() > 0: x[:, text_idx] = 0.0
            if dwell_idx is not None and dwell_idx.numel() > 0: x[:, dwell_idx] = 0.0
        elif variant == "base+text":
            if dwell_idx is not None and dwell_idx.numel() > 0: x[:, dwell_idx] = 0.0
        elif variant == "base+dwell":
            if text_idx is not None and text_idx.numel() > 0: x[:, text_idx] = 0.0
        elif variant == "base+text+dwell":
            pass


# ---- extra feature-group ablations (visit / accessibility / CBG socioeconomic) ----
# Resolved against the schema column names at runtime so we never zero the wrong columns.
_VISIT_POI_PREFIXES = ("raw_visit", "raw_visitor", "normalized_visits")
_ACCESS_CBG_NAMES = ("HealthAccessibility", "FoodAccessibility", "EduAccessibility",
                     "EntertainmentAccessibility", "Accessibility")
_CBG_KEEP_NONSOCIO = set(_ACCESS_CBG_NAMES) | {"longitude", "latitude", "Year"}

def _match_cols(names, exact=(), prefixes=()):
    idx = []
    for i, c in enumerate(names):
        c = str(c)
        if c in exact or any(c.startswith(p) for p in prefixes):
            idx.append(i)
    return idx

def _apply_feature_ablation_inplace(data, poi_names, cbg_names, ablation: str):
    """ablation ∈ {'none','no_visit','no_accessibility','no_cbg_socio'}.
    Zeros the matching feature columns (dimension unchanged). Aborts loudly if a
    requested group matches no columns, so an ablation can never silently no-op.
    Coordinates (longitude/latitude) are always preserved."""
    if ablation in (None, "", "none"):
        return
    if ablation == "no_visit":
        if poi_names is None:
            raise SystemExit("[abl] no_visit needs POI feature names (poi_features_json).")
        idx = _match_cols(poi_names, prefixes=_VISIT_POI_PREFIXES)
        if not idx:
            raise SystemExit(f"[abl] no_visit matched no POI columns. Have: {poi_names}")
        with torch.no_grad():
            data["poi"].x[:, torch.tensor(idx, dtype=torch.long, device=data['poi'].x.device)] = 0.0
        return
    if cbg_names is None:
        raise SystemExit(f"[abl] {ablation} needs CBG feature names (cbg_features_json).")
    if ablation == "no_accessibility":
        idx = _match_cols(cbg_names, exact=_ACCESS_CBG_NAMES)
        if not idx:
            raise SystemExit(f"[abl] no_accessibility matched no CBG columns. Have: {cbg_names}")
    elif ablation == "no_cbg_socio":
        idx = [i for i, c in enumerate(cbg_names) if str(c) not in _CBG_KEEP_NONSOCIO]
        if not idx:
            raise SystemExit(f"[abl] no_cbg_socio matched no CBG columns. Have: {cbg_names}")
    else:
        raise SystemExit(f"[abl] unknown feature_ablation '{ablation}'")
    with torch.no_grad():
        data["cbg"].x[:, torch.tensor(idx, dtype=torch.long, device=data['cbg'].x.device)] = 0.0


# =========================
# Model
# =========================
class MaybeGraphNorm(nn.Module):
    def __init__(self, dim: int, use: bool = True):
        super().__init__()
        self.use = use
        self.norm = GraphNorm(dim) if use else nn.Identity()
    def forward(self, x): return self.norm(x)

class VisitHeteroGNN(nn.Module):
    def __init__(
        self,
        poi_in_dim: int,
        cbg_in_dim: int,
        d_cbg: int = 256,
        d_poi: int = 128,
        d_hidden: int = 64,
        dropout: float = 0.15,
        include_visit_edges: bool = False,
        use_poi_poi: bool = True,
        use_graphnorm: bool = True,
        use_fallback: bool = True,   # <— ALWAYS ON in this script
        knn_use_attr: bool = True,   # if False, use unweighted conv for 'knn'
    ):
        super().__init__()
        self.dropout = dropout
        self.include_visit_edges = include_visit_edges
        self.use_poi_poi = use_poi_poi
        self.use_fallback = use_fallback
        self.knn_use_attr = knn_use_attr
        self.d_cbg, self.d_poi, self.d_hidden = d_cbg, d_poi, d_hidden

        # CBG 子图（2×SAGE 残差）
        self.cbg_proj  = Linear(cbg_in_dim, d_cbg, bias=False)
        self.cbg_conv1 = SAGEConv((d_cbg, d_cbg), d_cbg)
        self.cbg_conv2 = SAGEConv((d_cbg, d_cbg), d_cbg)
        self.cbg_norm1, self.cbg_norm2 = MaybeGraphNorm(d_cbg, use_graphnorm), MaybeGraphNorm(d_cbg, use_graphnorm)

        # POI 初始编码
        self.poi_mlp = nn.Sequential(Linear(poi_in_dim, d_poi), nn.ReLU(), Linear(d_poi, d_poi))
        self.poi_norm = MaybeGraphNorm(d_poi, use_graphnorm)

        # POI↔POI
        self.poi_poi_conv = None
        self.poi_poi_norm = MaybeGraphNorm(d_poi, use_graphnorm)

        # 跨类型
        self.cross_conv = None
        self.cross_norm = nn.ModuleDict({
            'cbg': MaybeGraphNorm(d_hidden, use_graphnorm),
            'poi': MaybeGraphNorm(d_hidden, use_graphnorm),
        })

        # Fallback 投影（always enabled）
        self.poi_to_hidden = Linear(d_poi, d_hidden, bias=False)
        self.cbg_to_hidden = Linear(d_cbg, d_hidden, bias=False)

        # Pair scorer
        self.pred_mlp = nn.Sequential(
            Linear(d_hidden * 2, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            Linear(d_hidden, d_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            Linear(d_hidden // 2, 32), nn.ReLU(), Linear(32, 1),
        )

    def _build_poi_poi_conv(self, edge_index_dict):
        if not self.use_poi_poi: return
        rels = {}
        spec = {'geo_knn': 2, 'time_sim': 1, 'brand': 1}
        for rel, edim in spec.items():
            et = ('poi', rel, 'poi')
            if et in edge_index_dict:
                rels[et] = GATv2Conv((self.d_poi, self.d_poi), self.d_poi,
                                     edge_dim=edim, add_self_loops=False)
        if rels:
            self.poi_poi_conv = HeteroConv(rels, aggr='sum').to(next(self.parameters()).device)

    def _build_cross_conv(self, edge_index_dict):
        rels = {}
        def _add(et, rel_name):
            s,_,d = et
            if rel_name == 'knn':
                if self.knn_use_attr:
                    # attribute-aware 'knn'
                    if s=='poi' and d=='cbg':
                        rels[et] = GATv2Conv((self.d_poi, self.d_cbg), self.d_hidden,
                                             edge_dim=1, add_self_loops=False)
                    elif s=='cbg' and d=='poi':
                        rels[et] = GATv2Conv((self.d_cbg, self.d_poi), self.d_hidden,
                                             edge_dim=1, add_self_loops=False)
                else:
                    # unweighted 'knn'
                    if s=='poi' and d=='cbg':
                        rels[et] = SAGEConv((self.d_poi, self.d_cbg), self.d_hidden)
                    elif s=='cbg' and d=='poi':
                        rels[et] = SAGEConv((self.d_cbg, self.d_poi), self.d_hidden)
            else:
                if s=='poi' and d=='cbg':
                    rels[et] = SAGEConv((self.d_poi, self.d_cbg), self.d_hidden)
                elif s=='cbg' and d=='poi':
                    rels[et] = SAGEConv((self.d_cbg, self.d_poi), self.d_hidden)

        allowed = ['belong', 'knn']
        if self.include_visit_edges:
            allowed.append('visit')  # 兼容开关（上层仍会过滤）

        for name in allowed:
            et = ('poi', name, 'cbg')
            if et in edge_index_dict: _add(et, name)
        for base in allowed:
            for rev in (f'rev_{base}', f'{base}__rev'):
                et = ('cbg', rev, 'poi')
                if et in edge_index_dict: _add(et, base)

        if rels:
            self.cross_conv = HeteroConv(rels, aggr='sum').to(next(self.parameters()).device)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # CBG 子图
        cbg_x = self.cbg_proj(x_dict['cbg'])
        if ('cbg', 'adjacent', 'cbg') in edge_index_dict:
            for conv, norm in ((self.cbg_conv1, self.cbg_norm1),
                               (self.cbg_conv2, self.cbg_norm2)):
                res = cbg_x
                cbg_x = conv((cbg_x, cbg_x), edge_index_dict[('cbg', 'adjacent', 'cbg')])
                cbg_x = F.relu(norm(cbg_x))
                cbg_x = F.dropout(cbg_x, self.dropout, self.training) + res

        # POI 初始
        poi_x = self.poi_mlp(x_dict['poi'])
        poi_x = F.relu(self.poi_norm(poi_x))
        poi_x = F.dropout(poi_x, self.dropout, self.training)

        # POI↔POI
        if self.use_poi_poi:
            if self.poi_poi_conv is None:
                self._build_poi_poi_conv(edge_index_dict)
            if self.poi_poi_conv is not None:
                z_poi = self.poi_poi_conv({'poi': poi_x}, edge_index_dict,
                                          edge_attr_dict=edge_attr_dict if edge_attr_dict else None)
                poi_x = F.relu(self.poi_poi_norm(poi_x + z_poi['poi']))
                poi_x = F.dropout(poi_x, self.dropout, self.training)

        # 跨类型
        if self.cross_conv is None:
            self._build_cross_conv(edge_index_dict)
        kwargs = {}
        if edge_attr_dict is not None:
            kwargs['edge_attr_dict'] = edge_attr_dict

        # fallback ON
        out = {'cbg': self.cbg_to_hidden(cbg_x), 'poi': self.poi_to_hidden(poi_x)}

        if self.cross_conv is not None:
            z = self.cross_conv({'cbg': cbg_x, 'poi': poi_x}, edge_index_dict, **kwargs)
            for key, val in z.items():
                out[key] = self.cross_norm[key](val)

        return out

    def predict_logits(self, z, knn_idx: torch.Tensor, invalid_mask: torch.Tensor):
        poi_emb, cbg_emb = z['poi'], z['cbg']
        N, K = knn_idx.shape
        filled = knn_idx.clone(); filled[invalid_mask] = 0
        cbg_knn = cbg_emb[filled]
        poi_rep = poi_emb.unsqueeze(1).expand(N, K, -1)
        pair = torch.cat([cbg_knn, poi_rep], dim=-1)
        logits = self.pred_mlp(pair.reshape(N * K, -1)).view(N, K)
        logits = logits.masked_fill(invalid_mask, float('-inf'))
        return logits

    def predict_probs(self, z, knn_idx: torch.Tensor, invalid_mask: torch.Tensor):
        logits = self.predict_logits(z, knn_idx, invalid_mask)
        probs = torch.softmax(logits, dim=1)
        return probs.masked_fill(invalid_mask, 0.0)


# =========================
# Targets / Masks / Metrics
# =========================
def build_targets_from_knn_candidates(data, K: int, device):
    e_knn = data[('poi', 'knn', 'cbg')]
    src = e_knn.edge_index[0].cpu().numpy()
    dst = e_knn.edge_index[1].cpu().numpy()
    if getattr(e_knn, 'edge_attr', None) is not None:
        dist = e_knn.edge_attr.view(-1).cpu().numpy()
        df = pd.DataFrame({'poi': src, 'cbg': dst, 'dist': dist}).sort_values(['poi', 'dist'])
    else:
        df = pd.DataFrame({'poi': src, 'cbg': dst}); df['dist'] = 0.0

    e_vis = data[('cbg', 'visit', 'poi')]
    v_src = e_vis.edge_index[0].cpu().numpy()
    v_dst = e_vis.edge_index[1].cpu().numpy()
    v_w   = e_vis.edge_attr.view(-1).cpu().numpy()
    visit_map = {(int(p), int(c)): float(w) for p, c, w in zip(v_dst, v_src, v_w)}

    num_poi = int(data['poi'].num_nodes)
    knn_idx = np.full((num_poi, K), -1, dtype=int)
    true_p  = np.zeros((num_poi, K), dtype=float)

    for poi, g in df.groupby('poi'):
        topk = g.head(K)
        cbgs = topk['cbg'].tolist()
        knn_idx[poi, :len(cbgs)] = cbgs
        ws = np.array([visit_map.get((poi, c), 0.0) for c in cbgs], dtype=float)
        s = ws.sum()
        true_p[poi, :len(cbgs)] = ws / (s + 1e-8) if s > 0 else 0.0

    idx_t = torch.tensor(knn_idx, device=device)
    true_t= torch.tensor(true_p, device=device, dtype=torch.float32)
    invalid = idx_t.lt(0)
    return idx_t, true_t, invalid

def masked_kl_loss(p: torch.Tensor, t: torch.Tensor, inv: torch.Tensor, eps=1e-8):
    mask = (~inv).float()
    p = p.clamp(min=eps); t = t.clamp(min=eps)
    kl = (t * (t.log() - p.log()) * mask).sum(dim=1)
    return (kl / mask.sum(dim=1).clamp(min=1.0)).mean()

@torch.no_grad()
def _rank_metrics(pred: torch.Tensor, true: torch.Tensor, inv: torch.Tensor):
    """
    Compute Top-1, MRR, NDCG for each row and then average.
    pred/true: [N, K]; inv: True where invalid.
    """
    N, K = pred.shape
    valid_row = (~inv).any(dim=1)
    if valid_row.sum() == 0:
        return {"top1": 0.0, "mrr": 0.0, "ndcg": 0.0}

    pr = pred.clone()
    tr = true.clone()
    pr[inv] = -1e9  # exclude invalid from ranking
    tr[inv] = 0.0

    top1_list, mrr_list, ndcg_list = [], [], []
    # positions 1..K -> log2(i+1)
    log_denom = torch.log2(torch.arange(2, K + 2, device=pred.device, dtype=pred.dtype))

    for i in torch.where(valid_row)[0].tolist():
        p = pr[i]; t = tr[i]
        if t.sum() <= 0:
            top1_list.append(0.0); mrr_list.append(0.0); ndcg_list.append(0.0); continue

        order = torch.argsort(p, descending=True)

        # Top-1 (argmax vs argmax)
        top1 = float(order[0].item() == torch.argmax(t).item())
        top1_list.append(top1)

        # MRR: first relevant position
        rel_mask = (t > 0)
        ranked_rel = rel_mask[order]
        if ranked_rel.any():
            rank = int(torch.where(ranked_rel)[0][0].item()) + 1  # 1-based
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        mrr_list.append(mrr)

        # NDCG with graded relevance
        gains = t[order]
        dcg = float((gains / log_denom).sum().item())
        ideal = torch.sort(t, descending=True)[0]
        idcg = float((ideal / log_denom).sum().item())
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_list.append(ndcg)

    return {
        "top1": float(np.mean(top1_list)),
        "mrr":  float(np.mean(mrr_list)),
        "ndcg": float(np.mean(ndcg_list)),
    }

@torch.no_grad()
def masked_metrics_with_ranks(pred, true, inv):
    # calibration-style
    kl  = masked_kl_loss(pred, true, inv).item()
    mae = (torch.abs(pred - true)[~inv]).mean().item()
    # r2
    y_true = true[~inv].detach().cpu().numpy()
    y_pred = pred[~inv].detach().cpu().numpy()
    try: r2 = r2_score(y_true, y_pred)
    except Exception: r2 = float('nan')
    # ranking-style
    rank = _rank_metrics(pred, true, inv)
    rank["kl"], rank["mae"], rank["r2"] = kl, mae, r2
    return rank


@torch.no_grad()
def report_metrics_like_tables(pred, true, inv, ndcg_k=50, recall_k=5, eps=1e-9):
    """EXACT replica of eval_on_intersection.metrics_on / evaluate_and_plot.metrics_basic.

    This is the function whose numbers MATCH the main results / cross-city tables:
      * each POI's pred & target are renormalised over its VALID candidates,
      * only POIs with a non-zero target are scored (gt>0 filter),
      * KL = sum_cand t*(log t - log p), summed over candidates (NOT divided by
        the candidate count) and then averaged over POIs  -> same scale as the
        main table (~0.30), unlike the training-objective KL (~0.006),
      * MAE is pooled (flat) over all valid candidates of all scored POIs,
      * Top-1 = hits / #scored POIs,
      * NDCG@k and Recall@k use the same definitions as the tables.
    mrr / r2 are also returned as extras (not in the main table but handy).
    Inputs pred/true/inv are [N, K] tensors already restricted to the eval split.
    """
    P = pred.detach().cpu().numpy()
    T = true.detach().cpu().numpy()
    INV = inv.detach().cpu().numpy()

    kls, abs_errs, ndcgs, recalls, mrrs = [], [], [], [], []
    r2_t, r2_p = [], []
    top1_hits, n = 0, 0

    for i in range(P.shape[0]):
        valid = ~INV[i]
        if valid.sum() == 0:
            continue
        p_raw = np.clip(P[i][valid].astype(float), 0.0, None)
        t_raw = np.clip(T[i][valid].astype(float), 0.0, None)
        if t_raw.sum() <= eps:                      # gt>0 filter
            continue
        sp = p_raw.sum()
        p = p_raw / sp if sp > eps else p_raw
        t = t_raw / t_raw.sum()
        n += 1
        # KL: sum over candidates, mean over POIs (NO division by candidate count)
        kls.append(float(np.sum(t * (np.log(t + eps) - np.log(p + eps)))))
        # MAE: pooled (flat) over all valid candidates
        abs_errs.extend(np.abs(p - t).tolist())
        # Top-1
        if int(np.argmax(p)) == int(np.argmax(t)):
            top1_hits += 1
        order = np.argsort(-p)
        # MRR (first relevant in predicted order)
        rel_in_order = (t[order] > 0)
        mrrs.append(1.0 / (int(np.argmax(rel_in_order)) + 1) if rel_in_order.any() else 0.0)
        # NDCG@k
        k = max(1, min(int(ndcg_k), len(p)))
        rel = t[order[:k]]
        dcg = float(np.sum(rel / np.log2(np.arange(2, 2 + len(rel)))))
        ideal = np.sort(t)[::-1][:k]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, 2 + len(ideal)))))
        if idcg > eps:
            ndcgs.append(dcg / idcg)
        # Recall@k = target mass captured by top-k predicted candidates
        rk = max(1, min(int(recall_k), len(p)))
        recalls.append(float(np.sum(t[order[:rk]])))
        r2_t.append(t); r2_p.append(p)

    if r2_t:
        yt = np.concatenate(r2_t); yp = np.concatenate(r2_p)
        try:
            r2 = float(r2_score(yt, yp))
        except Exception:
            r2 = float('nan')
    else:
        r2 = float('nan')

    return {
        "kl":     float(np.mean(kls)) if kls else float('nan'),
        "mae":    float(np.mean(abs_errs)) if abs_errs else float('nan'),
        "top1":   (top1_hits / n) if n else float('nan'),
        "ndcg":   float(np.mean(ndcgs)) if ndcgs else float('nan'),
        "recall": float(np.mean(recalls)) if recalls else float('nan'),
        "mrr":    float(np.mean(mrrs)) if mrrs else float('nan'),
        "r2":     r2,
        "n_poi":  n,
    }


# =========================
# Edge dict helpers
# =========================
POI_POI_REL_ALL = ('geo_knn','time_sim','brand')

def _is_visit(et):
    s, rel, d = et
    return rel in {'visit','rev_visit','visit__rev'}

def _base_rel(rel: str) -> str:
    # strip "rev_" prefix or "__rev" suffix to get base relation name
    if rel.startswith('rev_'): rel = rel[4:]
    if rel.endswith('__rev'): rel = rel[:-5]
    return rel

def _needs_edge_attr(et):
    s, rel, d = et
    if s=='poi' and d=='cbg' and _base_rel(rel)=='knn': return True
    if s=='cbg' and d=='poi' and _base_rel(rel)=='knn': return True
    if s=='poi' and d=='poi' and rel in POI_POI_REL_ALL: return True
    return False

def filter_edges(idx_dict: Dict, attr_dict: Dict,
                 keep_cbg_adj: bool = True,
                 poi_poi_modes: Tuple[str,...] = POI_POI_REL_ALL,
                 cross_modes: Tuple[str,...] = ('belong','knn'),
                 use_rev_edges: bool = True,
                 knn_use_attr: bool = True):
    def _keep(et):
        s, rel, d = et
        if _is_visit(et): return False
        if (s,d) == ('cbg','cbg'):
            return keep_cbg_adj and rel=='adjacent'
        if (s,d) == ('poi','poi'):
            return rel in poi_poi_modes
        if {s,d} == {'poi','cbg'}:
            base = _base_rel(rel)
            if base not in cross_modes: return False
            if (rel.startswith('rev_') or rel.endswith('__rev')) and (not use_rev_edges):
                return False
            return True
        return True

    out_idx  = {et: ei for et,ei in idx_dict.items()  if _keep(et)}
    # include only needed attrs; optionally drop knn attrs if knn_use_attr=False
    out_attr = {}
    for et, ea in attr_dict.items():
        if not _keep(et): continue
        if not _needs_edge_attr(et): continue
        if not knn_use_attr and (_base_rel(et[1]) == 'knn'):
            continue
        if ea is not None:
            out_attr[et] = ea
    return out_idx, out_attr


# ==============================
# Splits
# ==============================
def _safe_fetch_split(data, key: str):
    t = getattr(data, key, None)
    if isinstance(t, torch.Tensor): return t
    try:
        t = data[key]
        if isinstance(t, torch.Tensor): return t
    except Exception:
        pass
    if isinstance(data, dict):
        t = data.get(key, None)
        if isinstance(t, torch.Tensor): return t
    return None

def _get_splits(data, seed: int):
    tr = _safe_fetch_split(data, "train_idx")
    va = _safe_fetch_split(data, "val_idx")
    te = _safe_fetch_split(data, "test_idx")

    if (tr is not None) and (va is not None):
        if tr.dtype != torch.long: tr = tr.long()
        if va.dtype != torch.long: va = va.long()
        if (te is not None) and (te.dtype != torch.long): te = te.long()
        return tr.view(-1).cpu(), va.view(-1).cpu(), (te.view(-1).cpu() if te is not None else None)

    try:
        num_poi = int(getattr(data["poi"], "num_nodes", 0))
        if num_poi == 0:
            num_poi = int(data["poi"].x.size(0))
    except Exception:
        num_poi = int(data['poi'].x.size(0))

    rng = np.random.default_rng(seed)
    idx = np.arange(num_poi); rng.shuffle(idx)
    n_train = int(0.70 * num_poi)
    n_val   = int(0.15 * num_poi)
    train_idx = torch.tensor(idx[:n_train], dtype=torch.long)
    val_idx   = torch.tensor(idx[n_train:n_train + n_val], dtype=torch.long)
    test_idx  = torch.tensor(idx[n_train + n_val:], dtype=torch.long)
    return train_idx, val_idx, test_idx
def build_edge_index_for_train(data, keep_poi_idx: torch.Tensor):
    """
    Filter edges for training/eval based on a set of kept POI indices.
    - Removes self-loops for same-type edges.
    - Aligns edge_attr lengths to the filtered edge_index.
    - Keeps cross-type edges if at least the POI endpoint is kept.
    - For POI↔POI, keeps only edges with both endpoints kept.
    """
    mask = torch.zeros(data['poi'].num_nodes, dtype=torch.bool, device=keep_poi_idx.device)
    mask[keep_poi_idx] = True

    out_idx, out_attr = {}, {}
    for et in data.edge_types:
        s, rel, d = et
        ei = data[et].edge_index                      # [2, E]
        ea = getattr(data[et], 'edge_attr', None)     # [E, *] or [E_no_loop, *] or None

        E = ei.size(1)
        non_loop = torch.ones(E, dtype=torch.bool, device=ei.device)
        if s == d:  # drop self-loops only for homogeneous edges
            non_loop = (ei[0] != ei[1])

        # split-aware filtering
        if s == 'poi' and d == 'cbg':
            keep = mask[ei[0]] & non_loop
        elif s == 'cbg' and d == 'poi':
            keep = mask[ei[1]] & non_loop
        elif s == 'poi' and d == 'poi':
            keep = mask[ei[0]] & mask[ei[1]] & non_loop
        else:
            keep = non_loop

        out_idx[et] = ei[:, keep]

        # align edge_attr to filtered edges
        if ea is not None:
            if ea.size(0) == E:
                ea_sel = ea[keep]
            elif ea.size(0) == int(non_loop.sum().item()):
                # attributes were already computed after removing self-loops
                keep_in_nl = keep[non_loop]
                ea_sel = ea[keep_in_nl]
            else:
                # repair/pad/truncate to match either E or non_loop count
                target_len = int(non_loop.sum().item()) if (s == d) else E
                if ea.size(0) < target_len:
                    pad = torch.zeros((target_len - ea.size(0),) + ea.size()[1:],
                                      dtype=ea.dtype, device=ea.device)
                    ea_fix = torch.cat([ea, pad], dim=0)
                else:
                    ea_fix = ea[:target_len]
                if s == d:
                    keep_in_nl = keep[non_loop]
                    ea_sel = ea_fix[keep_in_nl]
                else:
                    ea_sel = ea_fix[keep]
            out_attr[et] = ea_sel

    return out_idx, out_attr


# =========================
# Train config
# =========================
@dataclass
class TrainConfig:
    graph_path: str
    out_dir: str = "./runs"
    # K decoupled
    k_train: int = 50
    k_eval: int = 50
    # reporting metric cutoffs (match the main results table)
    ndcg_k: int = 50
    recall_k: int = 5
    lr: float = 8.5e-4
    weight_decay: float = 0.0
    epochs: int = 1000
    seed: int = 42
    # dims
    d_cbg: int = 256
    d_poi: int = 128
    d_hidden: int = 64
    dropout: float = 0.10
    include_visit_edges: bool = False
    # arch toggles
    use_poi_poi: bool = True
    use_graphnorm: bool = True
    use_fallback: bool = True  # ALWAYS True here
    knn_use_attr: bool = True  # whether 'knn' uses attributes (GATv2) or unweighted (SAGE)
    # ablation toggles on graph
    keep_cbg_adj: bool = True
    poi_poi_modes: Tuple[str,...] = POI_POI_REL_ALL
    cross_modes: Tuple[str,...] = ('belong','knn')
    use_rev_edges: bool = True
    # features
    feature_variant: str = "base+text+dwell"
    feature_ablation: str = "none"   # none|no_visit|no_accessibility|no_cbg_socio
    poi_features_json: Optional[str] = None
    cbg_features_json: Optional[str] = None
    # misc
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_name: str = 'best.pt'
    params_json: Optional[str] = None

def update_cfg_from_json(cfg: TrainConfig, json_path: Optional[str]) -> TrainConfig:
    if not json_path: return cfg
    with open(json_path, "r") as f:
        best = json.load(f)
    for k, v in best.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# =========================
# Train / Eval single seed
# =========================
def run_once(cfg: TrainConfig, seed: int) -> Dict[str,float]:
    set_seed(seed)
    _safe_dir(cfg.out_dir)

    raw = torch.load(cfg.graph_path, weights_only=False)
    data = to_device_data(raw, cfg.device)

    # Feature variant（置零 text_/dwell_列）
    poi_json = cfg.poi_features_json or _auto_find_json(_dirname(cfg.graph_path), "poi")
    poi_names = _load_feature_names(poi_json)
    _apply_feature_variant_inplace(data, poi_names, variant=cfg.feature_variant)
    # Extra feature-group ablations (visit / accessibility / CBG socioeconomic)
    cbg_json = cfg.cbg_features_json or _auto_find_json(_dirname(cfg.graph_path), "cbg")
    cbg_names = _load_feature_names(cbg_json)
    _apply_feature_ablation_inplace(data, poi_names, cbg_names, cfg.feature_ablation)

    # splits
    train_idx, val_idx, test_idx = _get_splits(data, seed)
    train_idx = train_idx.to(cfg.device); val_idx = val_idx.to(cfg.device)
    test_idx  = (test_idx.to(cfg.device) if test_idx is not None else None)

    # KNN targets for TRAIN and EVAL (decoupled)
    knn_train, true_train, invalid_train = build_targets_from_knn_candidates(data, cfg.k_train, cfg.device)
    knn_eval,  true_eval,  invalid_eval  = build_targets_from_knn_candidates(data, cfg.k_eval,  cfg.device)

    # Build edge dicts
    all_poi = torch.arange(int(data['poi'].num_nodes), device=cfg.device)
    edge_full_idx, edge_full_attr_all = build_edge_index_for_train(data, all_poi)
    edge_train_idx, edge_train_attr_all = build_edge_index_for_train(data, train_idx)

    # Filter per ablation (incl. cross edges & knn attr toggle)
    edge_full_idx,  edge_full_attr  = filter_edges(
        edge_full_idx,  edge_full_attr_all,
        keep_cbg_adj=cfg.keep_cbg_adj,
        poi_poi_modes=cfg.poi_poi_modes if cfg.use_poi_poi else (),
        cross_modes=cfg.cross_modes,
        use_rev_edges=cfg.use_rev_edges,
        knn_use_attr=cfg.knn_use_attr
    )
    edge_train_idx, edge_train_attr = filter_edges(
        edge_train_idx, edge_train_attr_all,
        keep_cbg_adj=cfg.keep_cbg_adj,
        poi_poi_modes=cfg.poi_poi_modes if cfg.use_poi_poi else (),
        cross_modes=cfg.cross_modes,
        use_rev_edges=cfg.use_rev_edges,
        knn_use_attr=cfg.knn_use_attr
    )

    # Model & optim (fallback forced True)
    model = VisitHeteroGNN(
        int(data['poi'].x.size(1)), int(data['cbg'].x.size(1)),
        cfg.d_cbg, cfg.d_poi, cfg.d_hidden, cfg.dropout,
        include_visit_edges=cfg.include_visit_edges,
        use_poi_poi=cfg.use_poi_poi,
        use_graphnorm=cfg.use_graphnorm,
        use_fallback=True,
        knn_use_attr=cfg.knn_use_attr
    ).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state = float('inf'), None
    for ep in range(1, cfg.epochs+1):
        # train (optimize KL on TRAIN-K)
        model.train(); optim.zero_grad(set_to_none=True)
        z = model(data.x_dict, edge_train_idx, edge_attr_dict=edge_train_attr)
        p_train = model.predict_probs(z, knn_train, invalid_train)
        loss = masked_kl_loss(
            p_train.index_select(0, train_idx),
            true_train.index_select(0, train_idx),
            invalid_train.index_select(0, train_idx)
        )
        loss.backward(); optim.step()

        # validate on EVAL-K
        model.eval()
        with torch.no_grad():
            zv = model(data.x_dict, edge_full_idx, edge_attr_dict=edge_full_attr)
            pv = model.predict_probs(zv, knn_eval, invalid_eval)
            metr_va = masked_metrics_with_ranks(
                pv.index_select(0, val_idx),
                true_eval.index_select(0, val_idx),
                invalid_eval.index_select(0, val_idx)
            )
        val_kl = metr_va["kl"]
        if val_kl + 1e-6 < best_val:
            best_val, best_state = val_kl, model.state_dict()

    if best_state:
        torch.save(best_state, os.path.join(cfg.out_dir, cfg.save_name))
        model.load_state_dict(best_state)

    # test/eval（若无 test，用 val 代替），always on EVAL-K
    model.eval()
    with torch.no_grad():
        zt = model(data.x_dict, edge_full_idx, edge_attr_dict=edge_full_attr)
        pt = model.predict_probs(zt, knn_eval, invalid_eval)
    if test_idx is None:
        test_idx = val_idx
    metr_te = report_metrics_like_tables(
        pt.index_select(0, test_idx),
        true_eval.index_select(0, test_idx),
        invalid_eval.index_select(0, test_idx),
        ndcg_k=cfg.ndcg_k, recall_k=cfg.recall_k,
    )
    return metr_te


# =========================
# Suites (batch ablations)
# =========================
def suite_poi_poi(cfg_base: TrainConfig, seeds: List[int]):
    rows = []
    spec = [
        ("New_no_poi_poi", ()),
        ("New_geo_knn", ("geo_knn",)),
        ("New_time_sim", ("time_sim",)),
        ("New_brand", ("brand",)),
        ("New_all", POI_POI_REL_ALL),
    ]
    for name, modes in spec:
        cfg = TrainConfig(**{**cfg_base.__dict__})
        cfg.use_poi_poi = len(modes) > 0
        cfg.poi_poi_modes = modes
        cfg.save_name = f"{name}.pt"
        cfg.out_dir = os.path.join(cfg_base.out_dir, "poi_poi", name)
        for sd in seeds:
            metr = run_once(cfg, sd)
            rows.append(dict(
                suite="poi_poi", setting=name, seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    return rows

def suite_cbg_adj(cfg_base: TrainConfig, seeds: List[int]):
    rows = []
    for keep in (True, False):
        name = "with_cbg_adj" if keep else "no_cbg_adj"
        cfg = TrainConfig(**{**cfg_base.__dict__})
        cfg.keep_cbg_adj = keep
        cfg.save_name = f"{name}.pt"
        cfg.out_dir = os.path.join(cfg_base.out_dir, "cbg_adj", name)
        for sd in seeds:
            metr = run_once(cfg, sd)
            rows.append(dict(
                suite="cbg_adj", setting=name, seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    return rows

def suite_features(cfg_base: TrainConfig, seeds: List[int]):
    rows = []
    for fv in ("base","base+text","base+dwell","base+text+dwell"):
        cfg = TrainConfig(**{**cfg_base.__dict__})
        cfg.feature_variant = fv
        cfg.save_name = f"{fv}.pt"
        cfg.out_dir = os.path.join(cfg_base.out_dir, "features", fv.replace("+","_"))
        for sd in seeds:
            metr = run_once(cfg, sd)
            rows.append(dict(
                suite="features", setting=fv, seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    return rows

def suite_k(cfg_base: TrainConfig, seeds: List[int], k_values=(20,30,40,50,60,70,80)):
    rows = []
    for k in k_values:
        cfg = TrainConfig(**{**cfg_base.__dict__})
        cfg.k_train = int(k)   # vary training neighborhood
        cfg.save_name = f"ktrain{k}.pt"
        cfg.out_dir = os.path.join(cfg_base.out_dir, "K", f"ktrain_{k}")
        for sd in seeds:
            metr = run_once(cfg, sd)
            rows.append(dict(
                suite="K", setting=f"K_train={k}", seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    return rows

def suite_arch(cfg_base: TrainConfig, seeds: List[int]):
    rows = []
    spec = [
        ("default", True),        # GraphNorm ON
        ("no_graphnorm", False),  # GraphNorm OFF
    ]
    for name, use_gn in spec:
        cfg = TrainConfig(**{**cfg_base.__dict__})
        cfg.use_graphnorm = use_gn
        cfg.save_name = f"{name}.pt"
        cfg.out_dir = os.path.join(cfg_base.out_dir, "arch", name)
        for sd in seeds:
            metr = run_once(cfg, sd)
            rows.append(dict(
                suite="arch", setting=name, seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    return rows

def suite_cross(cfg_base: TrainConfig, seeds: List[int]):
    """
    Cross-type edges (POI↔CBG) ablation:
      - belong only
      - knn only
      - belong + knn
      - belong + knn (no reverse)
      - knn only, no attr (unweighted)
    """
    rows = []
    spec = [
        ("cross_belong_only", ('belong',), True,  True),
        ("cross_knn_only",    ('knn',),    True,  True),
        ("cross_both",        ('belong','knn'), True,  True),
        ("cross_both_no_rev", ('belong','knn'), False, True),
        ("cross_knn_no_attr", ('knn',),    True,  False),
    ]
    for name, modes, use_rev, knn_attr in spec:
        cfg = TrainConfig(**{**cfg_base.__dict__})
        cfg.cross_modes = modes
        cfg.use_rev_edges = use_rev
        cfg.knn_use_attr = knn_attr
        cfg.save_name = f"{name}.pt"
        cfg.out_dir = os.path.join(cfg_base.out_dir, "cross", name)
        for sd in seeds:
            metr = run_once(cfg, sd)
            rows.append(dict(
                suite="cross", setting=name, seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    return rows

def suite_feats_plus(cfg_base: TrainConfig, seeds: List[int]):
    """Extra feature-group ablations beyond text/dwell:
      full_ref          -- no extra zeroing (reference)
      no_visit          -- remove POI visit-derived features
      no_accessibility  -- remove the 5 engineered CBG accessibility features
      no_cbg_socio      -- remove CBG socioeconomic features (keep accessibility + coords)
    (Coordinates are always preserved.)"""
    rows = []
    for name in ("full_ref", "no_visit", "no_accessibility", "no_cbg_socio"):
        cfg = TrainConfig(**{**cfg_base.__dict__})
        cfg.feature_ablation = "none" if name == "full_ref" else name
        cfg.save_name = f"{name}.pt"
        cfg.out_dir = os.path.join(cfg_base.out_dir, "feats_plus", name)
        for sd in seeds:
            metr = run_once(cfg, sd)
            rows.append(dict(
                suite="feats_plus", setting=name, seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    return rows

SUITES = {
    "poi_poi": suite_poi_poi,
    "cbg_adj": suite_cbg_adj,
    "features": suite_features,
    "feats_plus": suite_feats_plus,
    "K": suite_k,
    "arch": suite_arch,
    "cross": suite_cross,
}


# =========================
# Aggregation helpers
# =========================
def _agg_mean_std_ci(df: pd.DataFrame, by=["suite","setting"]) -> pd.DataFrame:
    metrics = ["kl","mae","top1","mrr","ndcg","r2","recall"]
    metrics = [m for m in metrics if m in df.columns]
    gb = df.groupby(by, sort=False)
    out = gb[metrics].agg(['mean','std']).reset_index()
    # flatten cols
    out.columns = ['_'.join([c for c in col if c]).strip('_') for col in out.columns.values]
    # 95% CI half-width = 1.96 * std / sqrt(n), per group
    n = gb.size().reset_index(name='_n')
    out = out.merge(n, on=by, how='left')
    for m in metrics:
        std = out[f"{m}_std"].fillna(0.0)
        out[f"{m}_ci95"] = 1.96 * std / np.sqrt(out['_n'].clip(lower=1))
    out = out.drop(columns=['_n'])
    return out

def _with_deltas(df_agg: pd.DataFrame, by=["suite"]) -> pd.DataFrame:
    # deltas vs first row in each suite (keep ordering)
    def add_delta(g):
        ref = g.iloc[0]
        for m in ["kl_mean","mae_mean","top1_mean","mrr_mean","ndcg_mean","r2_mean","recall_mean"]:
            if m in g.columns:
                g[f"d_{m.replace('_mean','')}"] = g[m] - ref[m]
        return g
    frames = []
    for _, g in df_agg.groupby(by, sort=False):
        frames.append(add_delta(g.copy()))
    return pd.concat(frames, axis=0)


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_path", required=True)
    ap.add_argument("--out_dir", default="./runs")

    # optional feature name JSONs
    ap.add_argument("--poi_features_json", default=None)
    ap.add_argument("--cbg_features_json", default=None)

    # training basics
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)  # kept for compat; multi-seed uses --seeds
    ap.add_argument("--params_json", default=None)

    # K decoupled
    ap.add_argument("--k_train", type=int, default=50, help="Neighborhood size used for TRAIN objective.")
    ap.add_argument("--k_eval",  type=int, default=50, help="Fixed neighborhood size used for EVAL metrics.")
    # reporting cutoffs (match the main results table: NDCG@50, Recall@5)
    ap.add_argument("--ndcg_k",  type=int, default=50, help="NDCG cutoff for reported metrics.")
    ap.add_argument("--recall_k", type=int, default=5, help="Recall cutoff for reported metrics.")

    # which suites to run (comma-separated) or "single"
    ap.add_argument("--suite", default="poi_poi",
                    help="one or more of {poi_poi,cbg_adj,features,feats_plus,K,arch,cross} separated by comma; "
                         "use 'single' to run a single configuration defined by flags below")

    # single-run toggles (used when --suite=single)
    ap.add_argument("--lr", type=float, default=8.5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--d_cbg", type=int, default=256)
    ap.add_argument("--d_poi", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--use_poi_poi", action="store_true")
    ap.add_argument("--no_poi_poi", action="store_true")
    ap.add_argument("--keep_cbg_adj", action="store_true")
    ap.add_argument("--no_cbg_adj", action="store_true")
    ap.add_argument("--feature_variant", default="base+text+dwell",
                    choices=["base","base+text","base+dwell","base+text+dwell"])
    ap.add_argument("--feature_ablation", default="none",
                    choices=["none","no_visit","no_accessibility","no_cbg_socio"],
                    help="extra single-run feature-group ablation (suite 'feats_plus' runs all).")
    ap.add_argument("--use_graphnorm", action="store_true")
    ap.add_argument("--no_graphnorm", action="store_true")
    ap.add_argument("--save_name", default="best.pt")

    # cross-edge single-run controls (only for --suite=single)
    ap.add_argument("--cross_modes", default="belong,knn",
                    help="Comma-separated subset of {'belong','knn'} for POI↔CBG relations.")
    ap.add_argument("--no_rev_edges", action="store_true",
                    help="If set, drop reverse cross-type edges.")
    ap.add_argument("--knn_unweighted", action="store_true",
                    help="If set, treat 'knn' as unweighted (use SAGE instead of GATv2).")

    # multi-seed + K-suite values
    ap.add_argument("--seeds", default="41,42,43", help="Comma-separated seeds for multi-seed runs.")
    ap.add_argument("--k_values", default="20,30,40,50,60,70,80",
                    help="Comma-separated K_train values for suite K; k_eval stays fixed.")

    args = ap.parse_args()
    seeds = _parse_int_list(args.seeds)
    k_values = _parse_int_list(args.k_values)

    cfg = TrainConfig(
        graph_path=args.graph_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        seed=args.seed,
        poi_features_json=args.poi_features_json,
        cbg_features_json=args.cbg_features_json,
        save_name=args.save_name,
        k_train=args.k_train,
        k_eval=args.k_eval,
        ndcg_k=args.ndcg_k,
        recall_k=args.recall_k,
    )
    # JSON 覆盖
    cfg = update_cfg_from_json(cfg, args.params_json)

    # CLI 覆盖
    for key in ["lr","weight_decay","d_cbg","d_poi","d_hidden","dropout"]:
        if getattr(args, key, None) is not None:
            setattr(cfg, key, getattr(args, key))

    # edge/arch toggles
    if args.use_poi_poi: cfg.use_poi_poi = True
    if args.no_poi_poi:  cfg.use_poi_poi = False
    if args.keep_cbg_adj: cfg.keep_cbg_adj = True
    if args.no_cbg_adj:   cfg.keep_cbg_adj = False
    if args.use_graphnorm: cfg.use_graphnorm = True
    if args.no_graphnorm:  cfg.use_graphnorm = False
    cfg.feature_variant = args.feature_variant
    cfg.feature_ablation = args.feature_ablation

    # cross single-run toggles
    cfg.cross_modes = tuple([c.strip() for c in args.cross_modes.split(",") if c.strip()])
    cfg.use_rev_edges = not args.no_rev_edges
    cfg.knn_use_attr = not args.knn_unweighted

    # enforce fallback ON
    cfg.use_fallback = True

    suites = [s.strip() for s in args.suite.split(",")]
    all_rows = []

    if suites == ["single"]:
        # single config, multi-seed
        for sd in seeds:
            metr = run_once(cfg, sd)
            all_rows.append(dict(
                suite="single", setting="custom", seed=sd,
                k_train=cfg.k_train, k_eval=cfg.k_eval, **metr
            ))
    else:
        base = TrainConfig(**{**cfg.__dict__})
        # canonical defaults for base
        base.use_poi_poi = True
        base.poi_poi_modes = POI_POI_REL_ALL
        base.keep_cbg_adj = True
        base.use_graphnorm = True
        base.use_fallback = True
        base.feature_variant = cfg.feature_variant
        base.feature_ablation = "none"   # suites set their own; keep base clean
        base.k_eval = cfg.k_eval    # fixed eval K for all suites
        base.k_train = cfg.k_train  # default train K; suite K will override
        base.cross_modes = ('belong','knn')
        base.use_rev_edges = True
        base.knn_use_attr = True

        for s in suites:
            if s not in SUITES:
                print(f"[WARN] Unknown suite: {s}, skip."); continue
            if s == "K":
                rows = SUITES[s](base, seeds, k_values=tuple(k_values))
            else:
                rows = SUITES[s](base, seeds)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    _safe_dir(cfg.out_dir)

    per_seed_csv = os.path.join(cfg.out_dir, "ablation_per_seed.csv")
    df.to_csv(per_seed_csv, index=False)

    # Aggregates + deltas
    df_agg = _agg_mean_std_ci(df, by=["suite","setting"])
    df_agg = _with_deltas(df_agg, by=["suite"])
    agg_csv = os.path.join(cfg.out_dir, "ablation_aggregate.csv")
    df_agg.to_csv(agg_csv, index=False)

    # Pretty print
    show_cols = ["suite","setting",
                 "kl_mean","mae_mean","top1_mean","ndcg_mean","recall_mean","mrr_mean","r2_mean",
                 "kl_ci95","mae_ci95","top1_ci95","ndcg_ci95","recall_ci95"]
    show_cols = [c for c in show_cols if c in df_agg.columns]
    print("\n=== Ablation Results (aggregated, mean ± 95% CI) ===")
    disp = df_agg[show_cols].copy()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 180)
    print(disp.to_string(index=False))

    print(f"\n✓ Saved per-seed CSV → {per_seed_csv}")
    print(f"✓ Saved aggregate CSV → {agg_csv}")

if __name__ == "__main__":
    main()
    








# python /home/lp43319/projects/GNN/visitgnn/train_ablation.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --params_json /home/lp43319/projects/GNN/visitgnn/output/hyper/best_params.json \
#   --out_dir /home/lp43319/projects/GNN/visitgnn/output/ablation \
#   --suite poi_poi,cbg_adj,arch,cross \
#   --k_eval 50 \
#   --seeds 41,42,43 \
#   --epochs 1000



