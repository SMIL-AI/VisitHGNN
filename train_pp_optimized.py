#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, json, random, math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATv2Conv, HeteroConv, GraphNorm, Linear

# ==============================
# Utilities
# ==============================
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

def summarize_edges(title: str, idx: Dict, attr: Dict | None):
    print(f"\n=== {title} ===")
    for et in sorted(idx.keys(), key=lambda x:(x[0],x[1],x[2])):
        ei = idx[et]; ea = None if attr is None else attr.get(et, None)
        print(f"{str(et):30s}  E={ei.size(1):7d}  attr={None if ea is None else tuple(ea.size())}")
    print("===")

# ==============================
# Edge helpers (full-graph, aligned, no self-loops for homo-edges)
# ==============================
def build_edge_index_full(data: HeteroData):
    out_idx, out_attr = {}, {}
    for et in data.edge_types:
        s, rel, d = et
        ei = data[et].edge_index
        ea = getattr(data[et], 'edge_attr', None)
        E = ei.size(1)
        non_loop = torch.ones(E, dtype=torch.bool, device=ei.device)
        if s == d:
            non_loop = (ei[0] != ei[1])
        keep = non_loop
        out_idx[et] = ei[:, keep]
        if ea is not None:
            if ea.size(0) == E: ea_sel = ea[keep]
            elif ea.size(0) == int(non_loop.sum().item()):
                keep_in_nl = keep[non_loop]; ea_sel = ea[keep_in_nl]
            else:
                target_len = int(non_loop.sum().item()) if (s==d) else E
                if ea.size(0) < target_len:
                    pad = torch.zeros((target_len - ea.size(0),) + ea.size()[1:], dtype=ea.dtype, device=ea.device)
                    ea_fix = torch.cat([ea, pad], dim=0)
                else:
                    ea_fix = ea[:target_len]
                ea_sel = ea_fix[keep if s!=d else keep[non_loop]]
            out_attr[et] = ea_sel
    return out_idx, out_attr

POI_POI_ALL = ('geo_knn','time_sim','brand')

def filter_edges_for_use(idx: Dict, attr: Dict,
                         use_poi_poi: bool,
                         poi_poi_modes: Tuple[str,...],
                         keep_cbg_adj: bool,
                         include_visit_edges: bool,
                         cross_modes: Tuple[str,...] = ('belong','knn'),
                         use_rev_edges: bool = True):
    def is_visit(et):  return et[1] in {'visit','rev_visit','visit__rev'}
    def base_rel(rel): return rel[4:] if rel.startswith('rev_') else (rel[:-5] if rel.endswith('__rev') else rel)
    def keep_et(et):
        s, rel, d = et
        if is_visit(et) and (not include_visit_edges): return False
        if (s,d)==('cbg','cbg'): return keep_cbg_adj and (rel=='adjacent')
        if (s,d)==('poi','poi'):
            return use_poi_poi and (rel in poi_poi_modes)
        if {s,d}=={'poi','cbg'}:
            b = base_rel(rel)
            if b not in cross_modes: return False
            if (rel.startswith('rev_') or rel.endswith('__rev')) and (not use_rev_edges):
                return False
            return True
        return True

    o_idx  = {et:ei for et,ei in idx.items()  if keep_et(et)}
    o_attr = {}
    for et, ea in attr.items():
        if keep_et(et) and (ea is not None):
            o_attr[et] = ea
    return o_idx, o_attr

# ==============================
# Model
# ==============================
class Gate(nn.Module):
    """标量门控 α∈[0,1]，以 logit 形式学习；可设置初值。"""
    def __init__(self, a_init: float = 0.2):
        super().__init__()
        a = float(np.clip(a_init, 1e-3, 1-1e-3))
        logit = np.log(a/(1-a))
        self.logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
    def forward(self): 
        return torch.sigmoid(self.logit)

class VisitHeteroGNN(nn.Module):
    def __init__(self,
                 poi_in_dim: int,
                 cbg_in_dim: int,
                 d_cbg: int = 256,
                 d_poi: int = 128,
                 d_hidden: int = 64,
                 dropout: float = 0.10,
                 # toggles
                 use_poi_poi: bool = True,
                 poi_poi_modes: Tuple[str,...] = ('geo_knn','time_sim'),
                 include_visit_edges: bool = False,
                 aggr: str = 'mean',
                 # edge encoders
                 use_edge_norm: bool = True,
                 edge_temp: Dict[str,float] | None = None,
                 rel_scale: Dict[str,float] | None = None,
                 edge_mlp_dim: int = 4,
                 # learnables / gates
                 learnable_temp: bool = False,
                 learnable_rel_scale: bool = False,
                 use_rel_gates: bool = False,
                 gate_init: float = 0.1,
                 # anneal
                 ):
        super().__init__()
        self.dropout = dropout
        self.use_poi_poi = use_poi_poi
        self.poi_poi_modes = tuple(poi_poi_modes)
        self.include_visit_edges = include_visit_edges
        self.aggr = aggr
        self.use_edge_norm = use_edge_norm
        self.edge_temp = edge_temp or {'geo_knn':1.0,'time_sim':1.0,'brand':1.5}
        self.rel_scale = rel_scale or {'geo_knn':1.0,'time_sim':1.0,'brand':0.5}
        self.edge_mlp_dim = edge_mlp_dim
        self.learnable_temp = learnable_temp
        self.learnable_rel_scale = learnable_rel_scale
        self.use_rel_gates = use_rel_gates
        self.anneal_factor = 1.0  # 由外部训练循环设置（退火）

        self.cross_conv_attr  = None  # GATv2（带 edge_attr）
        self.cross_conv_plain = None  # SAGE（不带 edge_attr）
        self.cross_norm = nn.ModuleDict({'cbg': GraphNorm(d_hidden), 'poi': GraphNorm(d_hidden)})

        self.d_cbg, self.d_poi, self.d_hidden = d_cbg, d_poi, d_hidden

        # CBG encoder
        self.cbg_proj  = Linear(cbg_in_dim, d_cbg, bias=False)
        self.cbg_conv1 = SAGEConv((d_cbg, d_cbg), d_cbg)
        self.cbg_conv2 = SAGEConv((d_cbg, d_cbg), d_cbg)
        self.cbg_norm1, self.cbg_norm2 = GraphNorm(d_cbg), GraphNorm(d_cbg)

        # POI encoder
        self.poi_mlp = nn.Sequential(Linear(poi_in_dim, d_poi), nn.ReLU(), Linear(d_poi, d_poi))
        self.poi_norm = GraphNorm(d_poi)

        # POI-POI conv & edge encoders
        self.poi_poi_conv = None
        self.poi_poi_norm = GraphNorm(d_poi)
        self.gate = Gate(a_init=gate_init)  # 全局门控
        # 关系门控（可选）：对每个关系的边特征进行缩放
        if self.use_rel_gates:
            self.rel_gates = nn.ParameterDict({
                r: nn.Parameter(torch.tensor(0.0))  # 初值≈sigmoid(0)=0.5，稍后用sigmoid转[0,1]
                for r in POI_POI_ALL
            })
        else:
            self.rel_gates = None

        # learnable temp / scale（正值参数化）
        if self.learnable_temp:
            self.log_temp = nn.ParameterDict({
                r: nn.Parameter(torch.log(torch.tensor(self.edge_temp.get(r, 1.0), dtype=torch.float32)))
                for r in POI_POI_ALL
            })
        else:
            self.log_temp = None
        if self.learnable_rel_scale:
            self.log_scale = nn.ParameterDict({
                r: nn.Parameter(torch.log(torch.tensor(self.rel_scale.get(r, 1.0), dtype=torch.float32)))
                for r in POI_POI_ALL
            })
        else:
            self.log_scale = None

        # Fallback
        self.poi_to_hidden = Linear(d_poi, d_hidden, bias=False)
        self.cbg_to_hidden = Linear(d_cbg, d_hidden, bias=False)

        # Pair scorer
        self.pred_mlp = nn.Sequential(
            Linear(2*d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            Linear(d_hidden, d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            Linear(d_hidden//2, 32), nn.ReLU(), Linear(32, 1),
        )

        # relation-specific edge encoders (raw dim -> edge_mlp_dim)
        self.edge_mlps = nn.ModuleDict({
            'geo_knn': nn.Sequential(Linear(2, 8), nn.ReLU(), Linear(8, edge_mlp_dim)),
            'time_sim': nn.Sequential(Linear(1, 4), nn.ReLU(), Linear(4, edge_mlp_dim)),
            'brand':   nn.Sequential(Linear(1, 4), nn.ReLU(), Linear(4, edge_mlp_dim)),
        })

        # stats for online edge normalization (持久化，便于复现)
        for rel in POI_POI_ALL:
            self.register_buffer(f"{rel}_mean", torch.zeros(1), persistent=True)
            self.register_buffer(f"{rel}_std",  torch.ones(1),  persistent=True)

        # cache for analysis / regularization
        self.cache_ppoi_msg: Optional[torch.Tensor] = None  # 聚合后的 POI-POI 信息（用于 L2/日志）

    # ---------- builders ----------
    def _build_poi_poi_conv(self, edge_index_dict):
        if not self.use_poi_poi: return
        rels = {}
        for rel in self.poi_poi_modes:
            et = ('poi', rel, 'poi')
            if et in edge_index_dict:
                rels[et] = GATv2Conv((self.d_poi, self.d_poi), self.d_poi,
                                     edge_dim=self.edge_mlp_dim, add_self_loops=False)
        if rels:
            self.poi_poi_conv = HeteroConv(rels, aggr=self.aggr).to(next(self.parameters()).device)

    def _build_cross_conv(self, edge_index_dict):
        rels_attr, rels_plain = {}, {}

        def add(et, rel):
            s,_,d = et
            if rel == 'knn':
                if s=='poi' and d=='cbg':
                    rels_attr[et] = GATv2Conv((self.d_poi, self.d_cbg), self.d_hidden,
                                              edge_dim=1, add_self_loops=False)
                elif s=='cbg' and d=='poi':
                    rels_attr[et] = GATv2Conv((self.d_cbg, self.d_poi), self.d_hidden,
                                              edge_dim=1, add_self_loops=False)
            else:
                if s=='poi' and d=='cbg':
                    rels_plain[et] = SAGEConv((self.d_poi, self.d_cbg), self.d_hidden)
                elif s=='cbg' and d=='poi':
                    rels_plain[et] = SAGEConv((self.d_cbg, self.d_poi), self.d_hidden)

        allowed = ['belong','knn']
        for name in allowed:
            et = ('poi', name, 'cbg')
            if et in edge_index_dict: add(et, name)
        for base in allowed:
            for rev in (f'rev_{base}', f'{base}__rev'):
                et = ('cbg', rev, 'poi')
                if et in edge_index_dict: add(et, base)

        if rels_attr:
            self.cross_conv_attr  = HeteroConv(rels_attr,  aggr='sum').to(next(self.parameters()).device)
        if rels_plain:
            self.cross_conv_plain = HeteroConv(rels_plain, aggr='sum').to(next(self.parameters()).device)

    # ---------- anneal ----------
    def set_anneal(self, factor: float):
        self.anneal_factor = float(max(0.0, min(1.0, factor)))

    # ---------- edge attr prep ----------
    def _get_temp(self, rel: str) -> torch.Tensor:
        if self.learnable_temp and (self.log_temp is not None) and (rel in self.log_temp):
            return torch.exp(self.log_temp[rel]) + 1e-6
        return torch.tensor(self.edge_temp.get(rel, 1.0), device=next(self.parameters()).device)

    def _get_scale(self, rel: str) -> torch.Tensor:
        if self.learnable_rel_scale and (self.log_scale is not None) and (rel in self.log_scale):
            return torch.exp(self.log_scale[rel]) + 1e-6
        return torch.tensor(self.rel_scale.get(rel, 1.0), device=next(self.parameters()).device)

    def _prep_edge_attr(self, edge_attr_dict: Dict) -> Dict:
        """Normalize -> temperature -> relation scale -> relation gate(optional) -> MLP -> return new dict."""
        if edge_attr_dict is None: return {}
        out = {}
        for rel in self.poi_poi_modes:
            et = ('poi', rel, 'poi')
            ea = edge_attr_dict.get(et, None)
            if ea is None: continue
            x = ea
            # normalize
            if self.use_edge_norm and x.numel() > 0:
                m = getattr(self, f"{rel}_mean"); s = getattr(self, f"{rel}_std")
                if (s == 1).all() and (m == 0).all():
                    # lazily set stats from current batch（只设一次）
                    with torch.no_grad():
                        m_ = x.mean(dim=0, keepdim=True)
                        s_ = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
                        setattr(self, f"{rel}_mean", m_.detach())
                        setattr(self, f"{rel}_std",  s_.detach())
                        m, s = m_, s_
                x = (x - m) / s
            # temperature + relation scale
            tau  = self._get_temp(rel)
            scal = self._get_scale(rel)
            x = x / tau
            x = x * scal
            # relation gate（可选，sigmoid ∈ (0,1)）
            if self.rel_gates is not None:
                g = torch.sigmoid(self.rel_gates[rel])
                x = x * g
            # MLP to common dim
            x = self.edge_mlps[rel](x)
            out[et] = x
        # keep non-POI-POI edge_attr unchanged (e.g., knn 1-d weight)
        for et, ea in edge_attr_dict.items():
            if not (et[0]=='poi' and et[2]=='poi'):
                out[et] = ea
        return out

    # ---------- forward ----------
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, return_pp_msg: bool=False):
        # CBG
        cbg_x = self.cbg_proj(x_dict['cbg'])
        if ('cbg','adjacent','cbg') in edge_index_dict:
            for conv, norm in ((self.cbg_conv1, self.cbg_norm1),(self.cbg_conv2, self.cbg_norm2)):
                res = cbg_x
                cbg_x = conv((cbg_x, cbg_x), edge_index_dict[('cbg','adjacent','cbg')])
                cbg_x = F.relu(norm(cbg_x)); cbg_x = F.dropout(cbg_x, self.dropout, self.training) + res

        # POI initial
        poi_x = self.poi_mlp(x_dict['poi'])
        poi_x = F.relu(self.poi_norm(poi_x)); poi_x = F.dropout(poi_x, self.dropout, self.training)

        # POI-POI (gated residual)
        pp_msg = None
        if self.use_poi_poi:
            if self.poi_poi_conv is None: self._build_poi_poi_conv(edge_index_dict)
            if self.poi_poi_conv is not None:
                edge_attr_pp = self._prep_edge_attr(edge_attr_dict or {})
                z_pp = self.poi_poi_conv({'poi': poi_x}, edge_index_dict, edge_attr_dict=edge_attr_pp)
                pp_msg = z_pp['poi']  # 聚合后的 POI-POI 信息
                a = self.gate().clamp(0.0, 1.0) * self.anneal_factor
                poi_x = (1.0 - a) * poi_x + a * pp_msg
                poi_x = F.relu(self.poi_poi_norm(poi_x))
                poi_x = F.dropout(poi_x, self.dropout, self.training)

        self.cache_ppoi_msg = pp_msg  # 供正则/日志

        # Cross-type
        if self.cross_conv_attr is None and self.cross_conv_plain is None:
            self._build_cross_conv(edge_index_dict)

        out = {'cbg': self.cbg_to_hidden(cbg_x), 'poi': self.poi_to_hidden(poi_x)}  # fallback

        z_plain = self.cross_conv_plain({'cbg': cbg_x, 'poi': poi_x}, edge_index_dict) if self.cross_conv_plain else {}
        z_attr  = self.cross_conv_attr({'cbg': cbg_x, 'poi': poi_x}, edge_index_dict,
                                       edge_attr_dict=edge_attr_dict) if self.cross_conv_attr else {}

        for k in ('cbg','poi'):
            val = None
            if k in z_plain: val = z_plain[k] if val is None else (val + z_plain[k])
            if k in z_attr:  val = z_attr[k]  if val is None else (val + z_attr[k])
            if val is not None:
                out[k] = self.cross_norm[k](val)

        if return_pp_msg:
            return out, pp_msg
        return out

    # ---------- scoring ----------
    def predict_logits(self, z, knn_idx: torch.Tensor, invalid_mask: torch.Tensor):
        poi_e, cbg_e = z['poi'], z['cbg']
        N,K = knn_idx.shape
        filled = knn_idx.clone(); filled[invalid_mask] = 0
        cbg_knn = cbg_e[filled]; poi_rep = poi_e.unsqueeze(1).expand(N,K,-1)
        pair = torch.cat([cbg_knn, poi_rep], dim=-1)
        logits = self.pred_mlp(pair.reshape(N*K,-1)).view(N,K)
        return logits.masked_fill(invalid_mask, float('-inf'))

    def predict_probs(self, z, knn_idx: torch.Tensor, invalid_mask: torch.Tensor):
        return torch.softmax(self.predict_logits(z, knn_idx, invalid_mask), dim=1).masked_fill(invalid_mask, 0.0)

# ==============================
# Targets & metrics
# ==============================
def build_targets_from_knn_candidates(data, K: int, device):
    e_knn = data[('poi','knn','cbg')]
    src = e_knn.edge_index[0].cpu().numpy()
    dst = e_knn.edge_index[1].cpu().numpy()
    if getattr(e_knn, 'edge_attr', None) is not None:
        dist = e_knn.edge_attr.view(-1).cpu().numpy()
        df = __import__('pandas').DataFrame({'poi': src, 'cbg': dst, 'dist': dist}).sort_values(['poi','dist'])
    else:
        df = __import__('pandas').DataFrame({'poi': src, 'cbg': dst}); df['dist'] = 0.0

    e_vis = data[('cbg','visit','poi')]
    v_src = e_vis.edge_index[0].cpu().numpy()
    v_dst = e_vis.edge_index[1].cpu().numpy()
    v_w   = e_vis.edge_attr.view(-1).cpu().numpy()
    visit_map = {(int(p), int(c)): float(w) for p,c,w in zip(v_dst, v_src, v_w)}

    num_poi = int(data['poi'].num_nodes)
    knn_idx = np.full((num_poi, K), -1, dtype=int)
    true_p  = np.zeros((num_poi, K), dtype=float)
    for poi, g in df.groupby('poi'):
        topk = g.head(K); cbgs = topk['cbg'].tolist()
        knn_idx[poi,:len(cbgs)] = cbgs
        ws = np.array([visit_map.get((poi,c),0.0) for c in cbgs], dtype=float); s = ws.sum()
        true_p[poi,:len(cbgs)] = ws/(s+1e-8) if s>0 else 0.0
    idx_t = torch.tensor(knn_idx, device=device)
    true_t= torch.tensor(true_p,  device=device, dtype=torch.float32)
    invalid = idx_t.lt(0)
    return idx_t, true_t, invalid

def masked_kl_loss(p: torch.Tensor, t: torch.Tensor, inv: torch.Tensor, eps=1e-8):
    mask = (~inv).float()
    p = p.clamp(min=eps); t = t.clamp(min=eps)
    kl = (t * (t.log() - p.log()) * mask).sum(dim=1)
    return (kl / mask.sum(dim=1).clamp(min=1.0)).mean()

@torch.no_grad()
def masked_metrics(pred, true, inv):
    mask = ~inv
    kl  = masked_kl_loss(pred, true, inv).item()
    mae = (torch.abs(pred - true)[mask]).mean().item()
    pr = pred.clone(); pr[inv]=0; pr/=pr.sum(1,keepdim=True).clamp_min(1e-12)
    tr = true.clone(); tr[inv]=0; tr/=tr.sum(1,keepdim=True).clamp_min(1e-12)
    top1 = (pr.argmax(1)[mask.any(1)] == tr.argmax(1)[mask.any(1)]).float().mean().item()
    y_true = true[mask].cpu().numpy(); y_pred = pred[mask].cpu().numpy()
    try: r2 = r2_score(y_true, y_pred)
    except Exception: r2 = float('nan')
    return {"kl": kl, "mae": mae, "top1": top1, "r2": r2}

# ==============================
# Splits (deterministic fallback)
# ==============================
def _safe_fetch_split(data, key: str):
    t = getattr(data, key, None)
    if isinstance(t, torch.Tensor): return t
    try:
        t = data[key]; 
        if isinstance(t, torch.Tensor): return t
    except Exception: pass
    if isinstance(data, dict):
        t = data.get(key, None)
        if isinstance(t, torch.Tensor): return t
    return None

def get_splits(data, seed: int):
    tr = _safe_fetch_split(data, "train_idx")
    va = _safe_fetch_split(data, "val_idx")
    te = _safe_fetch_split(data, "test_idx")
    if (tr is not None) and (va is not None):
        tr = tr.long().view(-1).cpu(); va = va.long().view(-1).cpu()
        te = (te.long().view(-1).cpu() if te is not None else None)
        return tr, va, te
    rng = np.random.default_rng(seed)
    try: num_poi = int(getattr(data['poi'],'num_nodes',0)) or int(data['poi'].x.size(0))
    except Exception: num_poi = int(data['poi'].x.size(0))
    idx = np.arange(num_poi); rng.shuffle(idx)
    n_tr = int(num_poi*0.7); n_va = int(num_poi*0.15)
    return (torch.tensor(idx[:n_tr]), torch.tensor(idx[n_tr:n_tr+n_va]), torch.tensor(idx[n_tr+n_va:]))

# ==============================
# Train config & helpers
# ==============================
@dataclass
class TrainConfig:
    graph_path: str
    out_dir: str = "."
    seed: int = 42
    epochs: int = 800
    lr: float = 8.5e-4
    weight_decay: float = 0.0
    k: int = 50
    d_cbg: int = 256
    d_poi: int = 128
    d_hidden: int = 64
    dropout: float = 0.10
    include_visit_edges: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_name: str = "best_visit_gnn.pt"
    params_json: Optional[str] = None
    # POI-POI controls
    use_poi_poi: bool = True
    poi_poi_modes: Tuple[str,...] = ('geo_knn','time_sim')
    aggr: str = 'mean'
    use_edge_norm: bool = True
    edge_temp_geo: float = 1.0
    edge_temp_time: float = 1.0
    edge_temp_brand: float = 1.5
    rel_scale_geo: float = 1.0
    rel_scale_time: float = 1.0
    rel_scale_brand: float = 0.5
    edge_mlp_dim: int = 4
    # New: optimization toggles
    gate_init: float = 0.1
    learnable_temp: bool = False
    learnable_rel_scale: bool = False
    rel_gates: bool = False
    gate_anneal: str = "none"   # {"none","linear","cosine"}
    gate_anneal_warmup: int = 50
    pp_l2: float = 0.0          # L2 on POI-POI message
    # keep fallback
    use_fallback: bool = True

def update_cfg_from_json(cfg: TrainConfig, path: Optional[str]) -> TrainConfig:
    if not path: return cfg
    with open(path,'r',encoding='utf-8') as f:
        best = json.load(f)
    for k,v in best.items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    return cfg

def _anneal_factor(kind: str, epoch: int, warmup: int) -> float:
    if kind == "none": return 1.0
    if epoch <= warmup:
        t = epoch / max(1, warmup)
        if kind == "linear": return t
        if kind == "cosine": return 0.5 - 0.5*math.cos(math.pi*t)
    return 1.0

# ==============================
# Training
# ==============================
def train(cfg: TrainConfig):
    set_seed(cfg.seed); os.makedirs(cfg.out_dir, exist_ok=True)
    data = to_device_data(torch.load(cfg.graph_path, weights_only=False, map_location=cfg.device), cfg.device)

    # full graph edges once
    edge_idx_full, edge_attr_full = build_edge_index_full(data)
    edge_idx_full, edge_attr_full = filter_edges_for_use(
        edge_idx_full, edge_attr_full,
        use_poi_poi=cfg.use_poi_poi,
        poi_poi_modes=cfg.poi_poi_modes,
        keep_cbg_adj=True,
        include_visit_edges=cfg.include_visit_edges,
        cross_modes=('belong','knn'),
        use_rev_edges=True,
    )
    summarize_edges("FULL graph used (train/val/test)", edge_idx_full, edge_attr_full)

    # targets
    knn_idx, true_probs, invalid = build_targets_from_knn_candidates(data, cfg.k, cfg.device)

    # splits
    tr_idx, va_idx, te_idx = get_splits(data, cfg.seed)
    tr_idx, va_idx = tr_idx.to(cfg.device), va_idx.to(cfg.device)
    te_idx = (te_idx.to(cfg.device) if te_idx is not None else None)

    # model
    model = VisitHeteroGNN(
        int(data['poi'].x.size(1)), int(data['cbg'].x.size(1)),
        cfg.d_cbg, cfg.d_poi, cfg.d_hidden, cfg.dropout,
        use_poi_poi=cfg.use_poi_poi,
        poi_poi_modes=cfg.poi_poi_modes,
        include_visit_edges=cfg.include_visit_edges,
        aggr=cfg.aggr,
        use_edge_norm=cfg.use_edge_norm,
        edge_temp={'geo_knn':cfg.edge_temp_geo,'time_sim':cfg.edge_temp_time,'brand':cfg.edge_temp_brand},
        rel_scale={'geo_knn':cfg.rel_scale_geo,'time_sim':cfg.rel_scale_time,'brand':cfg.rel_scale_brand},
        edge_mlp_dim=cfg.edge_mlp_dim,
        learnable_temp=cfg.learnable_temp,
        learnable_rel_scale=cfg.learnable_rel_scale,
        use_rel_gates=cfg.rel_gates,
        gate_init=cfg.gate_init,
    ).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # training
    best_val, best_state = float('inf'), None
    hist = {k:[] for k in ["train_kl","val_kl","train_mae","val_mae","train_top1","val_top1"]}

    print(f"[Info] use_poi_poi={cfg.use_poi_poi}, modes={cfg.poi_poi_modes}, aggr={cfg.aggr}, edge_mlp_dim={cfg.edge_mlp_dim}")
    print(f"[Info] temps={{geo:{cfg.edge_temp_geo}, time:{cfg.edge_temp_time}, brand:{cfg.edge_temp_brand}}}, "
          f"rel_scale={{geo:{cfg.rel_scale_geo}, time:{cfg.rel_scale_time}, brand:{cfg.rel_scale_brand}}}")
    print(f"[Info] learnable_temp={cfg.learnable_temp}, learnable_rel_scale={cfg.learnable_rel_scale}, rel_gates={cfg.rel_gates}")
    print(f"[Info] gate_init={cfg.gate_init}, gate_anneal={cfg.gate_anneal}, warmup={cfg.gate_anneal_warmup}, pp_l2={cfg.pp_l2}")

    for ep in range(1, cfg.epochs+1):
        # 设置退火因子（只影响 forward 内的门控强度）
        af = _anneal_factor(cfg.gate_anneal, ep, cfg.gate_anneal_warmup)
        model.set_anneal(af)

        # train (full-graph; mask to train split)
        model.train(); optim.zero_grad(set_to_none=True)
        z, pp_msg = model(data.x_dict, edge_idx_full, edge_attr_dict=edge_attr_full, return_pp_msg=True)
        p = model.predict_probs(z, knn_idx, invalid)
        loss = masked_kl_loss(p.index_select(0, tr_idx), true_probs.index_select(0, tr_idx), invalid.index_select(0, tr_idx))
        # 额外的 POI-POI 分支 L2 正则（可选）
        if cfg.pp_l2 > 0.0 and (pp_msg is not None):
            loss = loss + cfg.pp_l2 * (pp_msg.pow(2).mean())
        loss.backward(); optim.step()

        # eval
        model.eval()
        with torch.no_grad():
            model.set_anneal(1.0)  # 验证统一用 full 强度（可按需换成 af）
            z_eval, pp_eval = model(data.x_dict, edge_idx_full, edge_attr_dict=edge_attr_full, return_pp_msg=True)
            p_eval = model.predict_probs(z_eval, knn_idx, invalid)
            metr_tr = masked_metrics(p_eval.index_select(0,tr_idx), true_probs.index_select(0,tr_idx), invalid.index_select(0,tr_idx))
            metr_va = masked_metrics(p_eval.index_select(0,va_idx), true_probs.index_select(0,va_idx), invalid.index_select(0,va_idx))

        for key,a,b in [("kl","train_kl","val_kl"),("mae","train_mae","val_mae"),("top1","train_top1","val_top1")]:
            hist[a].append(metr_tr[key]); hist[b].append(metr_va[key])

        if metr_va["kl"] + 1e-6 < best_val:
            best_val, best_state = metr_va["kl"], model.state_dict()
            torch.save(best_state, os.path.join(cfg.out_dir, cfg.save_name))

        # 日志：每 10 轮打印一次（含 gate α、pp_msg 强度）
        if ep==1 or ep%10==0 or ep==cfg.epochs:
            a_now = float(model.gate().item())
            pp_norm = float(pp_eval.norm(dim=1).mean().item()) if (pp_eval is not None) else 0.0
            print(f"Ep{ep:03d} | KL tr {metr_tr['kl']:.6f} va {metr_va['kl']:.6f} | "
                  f"MAE tr {metr_tr['mae']:.6f} va {metr_va['mae']:.6f} | "
                  f"T1 tr {metr_tr['top1']:.4f} va {metr_va['top1']:.4f} | "
                  f"gate α {a_now:.3f} (anneal {af:.2f}) | mean||pp_msg|| {pp_norm:.4f} | "
                  f"BestValKL {best_val:.6f}")

        # 诊断：每 50 轮测一次 on/off 对比强度
        if ep % 50 == 0 or ep==cfg.epochs:
            with torch.no_grad():
                # with poi-poi
                model.set_anneal(1.0)
                z_on = model(data.x_dict, edge_idx_full, edge_attr_dict=edge_attr_full)
                # temporarily disable
                old = model.poi_poi_conv; model.poi_poi_conv = None
                z_off = model(data.x_dict, edge_idx_full, edge_attr_dict=edge_attr_full)
                model.poi_poi_conv = old
                delta = (z_on['poi'] - z_off['poi']).norm(dim=1).mean().item()
                offn  = z_off['poi'].norm(dim=1).mean().item()
                ratio = (delta / max(offn, 1e-12))
                print(f"[Diag] Ep{ep:03d} mean||ΔPOI emb|| {delta:.6f} | mean||poi_emb_off|| {offn:.6f} | ratio {ratio:.3f}")

    if best_state: 
        model.load_state_dict(best_state)

    # curves
    try:
        hist = {k:np.array(v) for k,v in hist.items()}
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.plot(hist["train_kl"]);  plt.plot(hist["val_kl"]);  plt.title("KL");  plt.legend(["Train","Val"])
        plt.subplot(1,3,2); plt.plot(hist["train_mae"]); plt.plot(hist["val_mae"]); plt.title("MAE")
        plt.subplot(1,3,3); plt.plot(hist["train_top1"]);plt.plot(hist["val_top1"]);plt.title("Top-1")
        plt.tight_layout(); plt.savefig(os.path.join(cfg.out_dir, "training_curves.png")); plt.close()
    except Exception: pass

    # test
    if te_idx is not None:
        model.eval()
        with torch.no_grad():
            z_test = model(data.x_dict, edge_idx_full, edge_attr_dict=edge_attr_full)
            p_test = model.predict_probs(z_test, knn_idx, invalid)
            metr_te = masked_metrics(p_test.index_select(0,te_idx), true_probs.index_select(0,te_idx), invalid.index_select(0,te_idx))
        print("Final TEST | KL {:.6f}  MAE {:.6f}  Top-1 {:.4f}  R2 {:.4f}".format(
            metr_te["kl"], metr_te["mae"], metr_te["top1"], metr_te["r2"]))

    # 自检：POI-POI 影响强度
    if cfg.use_poi_poi:
        with torch.no_grad():
            z_on = model(data.x_dict, edge_idx_full, edge_attr_dict=edge_attr_full)
            old = model.poi_poi_conv; model.poi_poi_conv = None
            z_off = model(data.x_dict, edge_idx_full, edge_attr_dict=edge_attr_full)
            model.poi_poi_conv = old
            diff = (z_on['poi'] - z_off['poi']).norm(dim=1).mean().item()
            offn = z_off['poi'].norm(dim=1).mean().item()
            ratio = diff / max(offn, 1e-12)
            print(f"[Sanity] mean ||ΔPOI emb||: {diff:.6f} | mean||poi_emb_off||: {offn:.6f} | ratio: {ratio:.3f}")

# ==============================
# CLI
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_path", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=8.5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--d_cbg", type=int, default=256)
    ap.add_argument("--d_poi", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--params_json", default=None)
    ap.add_argument("--save_name", default="best_visit_gnn.pt")
    # POI-POI toggles
    ap.add_argument("--use_poi_poi", dest="use_poi_poi", action="store_true")
    ap.add_argument("--no_poi_poi",  dest="use_poi_poi", action="store_false")
    ap.set_defaults(use_poi_poi=True)
    ap.add_argument("--poi_poi_modes", default="geo_knn,time_sim",
                    help="subset of {geo_knn,time_sim,brand} separated by comma")
    ap.add_argument("--aggr", default="mean", choices=["sum","mean","max"])
    ap.add_argument("--no_edge_norm", action="store_true")
    ap.add_argument("--edge_mlp_dim", type=int, default=4)
    ap.add_argument("--edge_temp_geo", type=float, default=1.0)
    ap.add_argument("--edge_temp_time", type=float, default=1.0)
    ap.add_argument("--edge_temp_brand", type=float, default=1.5)
    ap.add_argument("--rel_scale_geo", type=float, default=1.0)
    ap.add_argument("--rel_scale_time", type=float, default=1.0)
    ap.add_argument("--rel_scale_brand", type=float, default=0.5)
    # New: 优化开关
    ap.add_argument("--gate_init", type=float, default=0.1)
    ap.add_argument("--learnable_temp", action="store_true")
    ap.add_argument("--learnable_rel_scale", action="store_true")
    ap.add_argument("--rel_gates", action="store_true")
    ap.add_argument("--gate_anneal", choices=["none","linear","cosine"], default="none")
    ap.add_argument("--gate_anneal_warmup", type=int, default=50)
    ap.add_argument("--pp_l2", type=float, default=0.0)
    ap.add_argument("--no_fallback", dest="use_fallback", action="store_false")
    ap.set_defaults(use_fallback=True)

    args = ap.parse_args()

    cfg = TrainConfig(
        graph_path=args.graph_path, out_dir=args.out_dir, seed=args.seed,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, k=args.k,
        d_cbg=args.d_cbg, d_poi=args.d_poi, d_hidden=args.d_hidden, dropout=args.dropout,
        params_json=args.params_json, save_name=args.save_name,
        use_poi_poi=args.use_poi_poi,
        poi_poi_modes=tuple([s.strip() for s in args.poi_poi_modes.split(",") if s.strip()]),
        aggr=args.aggr, use_edge_norm=(not args.no_edge_norm),
        edge_mlp_dim=args.edge_mlp_dim,
        edge_temp_geo=args.edge_temp_geo, edge_temp_time=args.edge_temp_time, edge_temp_brand=args.edge_temp_brand,
        rel_scale_geo=args.rel_scale_geo, rel_scale_time=args.rel_scale_time, rel_scale_brand=args.rel_scale_brand,
        gate_init=args.gate_init,
        learnable_temp=args.learnable_temp,
        learnable_rel_scale=args.learnable_rel_scale,
        rel_gates=args.rel_gates,
        gate_anneal=args.gate_anneal,
        gate_anneal_warmup=args.gate_anneal_warmup,
        pp_l2=args.pp_l2,
        use_fallback=args.use_fallback,
    )
    cfg = update_cfg_from_json(cfg, args.params_json)
    train(cfg)

if __name__ == "__main__":
    main()





# python train_pp_optimized.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --out_dir /home/lp43319/projects/GNN/visitgnn/output/Train_pp_optimized \
#   --epochs 1000 --seed 42 \
# --gate_init 0.12 \
# --gate_anneal cosine --gate_anneal_warmup 80 \
# --learnable_rel_scale --rel_gates \
# --edge_temp_geo 1.2 --edge_temp_time 1.2 \
# --pp_l2 5e-6
# --dropout 0.15 --weight_decay 3e-5
