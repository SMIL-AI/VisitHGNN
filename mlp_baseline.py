
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature-only baseline: two-tower MLP for POI-CBG ranking.
- No message passing or graph convolution.
- Builds KNN candidate sets from ('poi','knn','cbg'); optionally uses the distance as an input feature.
- Optimizes KL divergence (cross-entropy with soft labels) within K per-POI candidate set.
- Reports Top-1, MRR, NDCG, KL, MAE, R^2 and saves predictions.
"""
from __future__ import annotations

import argparse, os, json, math, random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

# -----------------------------
# Repro / utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_of(t: torch.Tensor) -> torch.device:
    return t.device if isinstance(t, torch.Tensor) else torch.device("cpu")

def to_device_data(obj, device):
    # Accept HeteroData or {'graph_data': HeteroData}
    if isinstance(obj, dict) and "graph_data" in obj:
        obj = obj["graph_data"]
    return obj.to(device)

def safe_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# -----------------------------
# Feature helpers (match your ablation style)
# -----------------------------
def auto_find_json(dirpath: str, kind: str) -> Optional[str]:
    import glob
    patt = os.path.join(dirpath, f"*_{kind}_feature_names.json")
    cands = sorted(glob.glob(patt))
    return cands[-1] if cands else None

def load_feature_names(json_path: Optional[str]) -> Optional[List[str]]:
    if not json_path or (not os.path.isfile(json_path)): return None
    with open(json_path, "r") as f:
        obj = json.load(f)
    cols = obj.get("columns", None)
    if isinstance(cols, list): return [str(c) for c in cols]
    return None

def build_masks_from_names(names: Optional[List[str]]):
    if not names: return None, None
    text_idx = [i for i, c in enumerate(names) if str(c).startswith("text_")]
    dwell_idx = [i for i, c in enumerate(names) if str(c).startswith("dwell_")]
    return text_idx, dwell_idx

def apply_feature_variant_inplace(x: torch.Tensor,
                                  poi_names: Optional[List[str]],
                                  variant: str = "base+dwell"):
    """
    variant in {"base","base+text","base+dwell","base+text+dwell"}.
    We zero the columns to preserve the schema/dims.
    """
    if poi_names is None: return
    text_idx, dwell_idx = build_masks_from_names(poi_names)
    if variant not in {"base","base+text","base+dwell","base+text+dwell"}:
        return
    with torch.no_grad():
        if variant == "base":
            if text_idx:  x[:, torch.tensor(text_idx, device=x.device)] = 0.0
            if dwell_idx: x[:, torch.tensor(dwell_idx, device=x.device)] = 0.0
        elif variant == "base+text":
            if dwell_idx: x[:, torch.tensor(dwell_idx, device=x.device)] = 0.0
        elif variant == "base+dwell":
            if text_idx:  x[:, torch.tensor(text_idx, device=x.device)] = 0.0
        elif variant == "base+text+dwell":
            pass

# -----------------------------
# Build KNN candidates & true probs
# -----------------------------
@torch.no_grad()
def build_knn_and_distances(data, K: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (knn_idx [N,K], invalid_mask [N,K], dist [N,K]) sorted by distance if available."""
    e = data[('poi','knn','cbg')]
    src = e.edge_index[0].cpu().numpy()
    dst = e.edge_index[1].cpu().numpy()
    if getattr(e, 'edge_attr', None) is not None:
        dist = e.edge_attr.view(-1).cpu().numpy()
        df = pd.DataFrame({'poi': src, 'cbg': dst, 'dist': dist}).sort_values(['poi','dist'])
    else:
        df = pd.DataFrame({'poi': src, 'cbg': dst}); df['dist'] = 0.0

    N = int(data['poi'].num_nodes)
    knn = np.full((N, K), -1, dtype=np.int64)
    dd  = np.full((N, K), np.nan, dtype=np.float32)
    for poi, g in df.groupby('poi'):
        topk = g.head(K)
        cbgs = topk['cbg'].astype(int).to_numpy()
        dvec = topk['dist'].astype(float).to_numpy()
        knn[poi, :len(cbgs)] = cbgs
        dd[poi, :len(dvec)]  = dvec
    knn_t = torch.tensor(knn, device=device, dtype=torch.long)
    dist_t= torch.tensor(np.nan_to_num(dd, nan=0.0), device=device, dtype=torch.float32)
    invalid = knn_t.lt(0)
    return knn_t, invalid, dist_t

@torch.no_grad()
def build_true_probs_from_graph(data, knn_idx: torch.Tensor, device) -> torch.Tensor:
    """Ground-truth probability over the K candidates using ('cbg','visit','poi') weights."""
    e = data[('cbg','visit','poi')]
    cbg = e.edge_index[0].cpu().numpy()
    poi = e.edge_index[1].cpu().numpy()
    w   = e.edge_attr.view(-1).cpu().numpy()
    visit_map = {(int(p), int(c)): float(wt) for p, c, wt in zip(poi, cbg, w)}
    N, K = knn_idx.shape
    t = torch.zeros((N, K), dtype=torch.float32, device=device)
    for i in range(N):
        for j in range(K):
            c = int(knn_idx[i, j].item())
            if c >= 0:
                t[i, j] = visit_map.get((i, c), 0.0)
    row_sum = t.sum(1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return t / row_sum

# -----------------------------
# Splits
# -----------------------------
def get_splits(data, seed: int = 42):
    def _fetch(key):
        t = getattr(data, key, None)
        if isinstance(t, torch.Tensor): return t
        try:
            t = data[key]
        except Exception:
            t = None
        return t if isinstance(t, torch.Tensor) else None

    tr, va, te = _fetch("train_idx"), _fetch("val_idx"), _fetch("test_idx")
    if tr is not None and va is not None:
        return tr.view(-1).long().cpu(), va.view(-1).long().cpu(), (te.view(-1).long().cpu() if te is not None else None)

    # fallback 70/15/15
    rng = np.random.default_rng(seed)
    N = int(data['poi'].num_nodes)
    order = np.arange(N); rng.shuffle(order)
    n_tr = int(0.70 * N); n_va = int(0.15 * N)
    tr = torch.tensor(order[:n_tr]); va = torch.tensor(order[n_tr:n_tr+n_va]); te = torch.tensor(order[n_tr+n_va:])
    return tr, va, te

# -----------------------------
# Metrics
# -----------------------------
def masked_kl_loss(p: torch.Tensor, t: torch.Tensor, inv: torch.Tensor, eps=1e-8):
    mask = (~inv).float()
    p = p.clamp(min=eps); t = t.clamp(min=eps)
    kl = (t * (t.log() - p.log()) * mask).sum(dim=1)
    return (kl / mask.sum(dim=1).clamp(min=1.0)).mean()

@torch.no_grad()
def rank_metrics(pred: torch.Tensor, true: torch.Tensor, inv: torch.Tensor):
    N, K = pred.shape
    valid_row = (~inv).any(1)
    if valid_row.sum() == 0:
        return {"top1": 0.0, "mrr": 0.0, "ndcg": 0.0}
    pr = pred.clone(); tr = true.clone()
    pr[inv] = -1e9; tr[inv] = 0.0
    log_denom = torch.log2(torch.arange(2, K+2, device=pred.device, dtype=pred.dtype))

    top1_list, mrr_list, ndcg_list = [], [], []
    for i in torch.where(valid_row)[0].tolist():
        p = pr[i]; t = tr[i]
        if t.sum() <= 0:
            top1_list.append(0.0); mrr_list.append(0.0); ndcg_list.append(0.0); continue
        order = torch.argsort(p, descending=True)
        top1_list.append(float(order[0].item() == torch.argmax(t).item()))
        rel_mask = (t > 0); ranked_rel = rel_mask[order]
        if ranked_rel.any():
            rank = int(torch.where(ranked_rel)[0][0].item()) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        mrr_list.append(mrr)
        gains = t[order]
        dcg = float((gains / log_denom).sum().item())
        ideal = torch.sort(t, descending=True)[0]
        idcg = float((ideal / log_denom).sum().item())
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_list.append(ndcg)
    return {"top1": float(np.mean(top1_list)), "mrr": float(np.mean(mrr_list)), "ndcg": float(np.mean(ndcg_list))}

@torch.no_grad()
def full_metrics(pred: torch.Tensor, true: torch.Tensor, inv: torch.Tensor):
    kl  = masked_kl_loss(pred, true, inv).item()
    mae = (torch.abs(pred - true)[~inv]).mean().item()
    rank = rank_metrics(pred, true, inv)
    # R^2 over valid positions after per-row normalization
    p = pred.clone(); p[inv] = 0; p = p / p.sum(1, keepdim=True).clamp(min=1e-12)
    t = true.clone(); t[inv] = 0; t = t / t.sum(1, keepdim=True).clamp(min=1e-12)
    y_true = t[~inv].detach().cpu().numpy()
    y_pred = p[~inv].detach().cpu().numpy()
    try: r2 = r2_score(y_true, y_pred)
    except Exception: r2 = float('nan')
    return {"kl": kl, "mae": mae, "top1": rank["top1"], "mrr": rank["mrr"], "ndcg": rank["ndcg"], "r2": r2}

# -----------------------------
# Model
# -----------------------------
class TwoTowerMLP(nn.Module):
    def __init__(self, poi_in: int, cbg_in: int,
                 poi_hidden: int = 128, cbg_hidden: int = 256,
                 pair_hidden: int = 64, dropout: float = 0.10,
                 use_distance: bool = True):
        super().__init__()
        self.use_distance = use_distance
        self.poi = nn.Sequential(
            nn.Linear(poi_in, poi_hidden), nn.ReLU(),
            nn.Linear(poi_hidden, poi_hidden), nn.ReLU(),
        )
        self.cbg = nn.Sequential(
            nn.Linear(cbg_in, cbg_hidden), nn.ReLU(),
            nn.Linear(cbg_hidden, cbg_hidden), nn.ReLU(),
        )
        pair_in = poi_hidden + cbg_hidden + (1 if use_distance else 0)
        self.head = nn.Sequential(
            nn.Linear(pair_in, pair_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(pair_hidden, pair_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(pair_hidden, 1),
        )

    def forward_logits(self, poi_x: torch.Tensor, cbg_x: torch.Tensor,
                       knn_idx: torch.Tensor, invalid: torch.Tensor,
                       dist: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        poi_x: [N_poi, Dp]; cbg_x: [N_cbg, Dc]
        knn_idx: [B, K]; invalid: [B, K]; dist: [B, K] or None
        Returns: logits [B, K] with -inf at invalid positions.
        """
        B, K = knn_idx.shape
        poi_emb = self.poi(poi_x)                # [N_poi, Hp]
        cbg_emb = self.cbg(cbg_x)                # [N_cbg, Hc]
        # gather
        cbg_knn = cbg_emb[knn_idx.clamp_min(0)]  # [B, K, Hc]
        # broadcast poi
        poi_ids = torch.arange(B, device=knn_idx.device)
        poi_rep = poi_emb[poi_ids].unsqueeze(1).expand(B, K, -1)  # [B, K, Hp]
        feats = [poi_rep, cbg_knn]
        if self.use_distance and (dist is not None):
            feats.append(dist.unsqueeze(-1))  # [B, K, 1]
        pair = torch.cat(feats, dim=-1)       # [B, K, Hp+Hc(+1)]
        logits = self.head(pair.reshape(B*K, -1)).view(B, K)
        logits = logits.masked_fill(invalid, float('-inf'))
        return logits

    @torch.no_grad()
    def predict_probs(self, poi_x, cbg_x, knn_idx, invalid, dist=None) -> torch.Tensor:
        logits = self.forward_logits(poi_x, cbg_x, knn_idx, invalid, dist)
        probs = torch.softmax(logits, dim=1)
        return probs.masked_fill(invalid, 0.0)

# -----------------------------
# Train loop
# -----------------------------
@dataclass
class Config:
    graph_path: str
    out_dir: str = "./mlp_baseline_outputs"
    k_train: int = 40
    k_eval: int  = 40
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2048
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # model dims
    poi_hidden: int = 128
    cbg_hidden: int = 256
    pair_hidden: int = 64
    dropout: float = 0.10
    use_distance: bool = True
    # feature variants
    feature_variant: str = "base+dwell"
    poi_features_json: Optional[str] = None
    cbg_features_json: Optional[str] = None

def run(cfg: Config):
    set_seed(cfg.seed)
    safe_dir(cfg.out_dir)

    data = torch.load(cfg.graph_path, weights_only=False)
    data = to_device_data(data, cfg.device)

    # Feature variants (POI side only; CBG untouched)
    poi_json = cfg.poi_features_json or auto_find_json(os.path.dirname(cfg.graph_path), "poi")
    poi_names = load_feature_names(poi_json)
    apply_feature_variant_inplace(data['poi'].x, poi_names, variant=cfg.feature_variant)

    # Splits
    train_idx, val_idx, test_idx = get_splits(data, seed=cfg.seed)
    train_idx = train_idx.to(cfg.device); val_idx = val_idx.to(cfg.device)
    test_idx  = (test_idx.to(cfg.device) if test_idx is not None else None)

    # KNN candidates & distances + GT for both TRAIN-K and EVAL-K
    knn_train, inv_train, dist_train = build_knn_and_distances(data, cfg.k_train, cfg.device)
    knn_eval,  inv_eval,  dist_eval  = build_knn_and_distances(data, cfg.k_eval,  cfg.device)
    true_train = build_true_probs_from_graph(data, knn_train, cfg.device)
    true_eval  = build_true_probs_from_graph(data, knn_eval,  cfg.device)

    # Model
    poi_in = int(data['poi'].x.size(1)); cbg_in = int(data['cbg'].x.size(1))
    model = TwoTowerMLP(poi_in, cbg_in, cfg.poi_hidden, cfg.cbg_hidden, cfg.pair_hidden,
                        cfg.dropout, use_distance=cfg.use_distance).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Helper to run a split in mini-batches (by POI rows)
    def run_epoch(split: str, grad: bool):
        if split == "train":
            poi_rows = train_idx
            knn, inv, dist, true = knn_train, inv_train, dist_train, true_train
        elif split == "val":
            poi_rows = val_idx
            knn, inv, dist, true = knn_eval, inv_eval, dist_eval, true_eval
        else:
            poi_rows = test_idx if test_idx is not None else val_idx
            knn, inv, dist, true = knn_eval, inv_eval, dist_eval, true_eval

        total_loss = 0.0
        all_probs = []
        B = poi_rows.numel()
        bs = cfg.batch_size
        for start in range(0, B, bs):
            end = min(B, start + bs)
            rows = poi_rows[start:end]
            # slice candidate rows
            k_idx = knn.index_select(0, rows)
            inv_m = inv.index_select(0, rows)
            d_sub = dist.index_select(0, rows) if cfg.use_distance else None
            # forward
            logits = model.forward_logits(data['poi'].x, data['cbg'].x, k_idx, inv_m, d_sub)
            probs = torch.softmax(logits, dim=1).masked_fill(inv_m, 0.0)
            if grad:
                t = true.index_select(0, rows)
                loss = masked_kl_loss(probs, t, inv_m)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                total_loss += float(loss.item()) * (end - start)
            else:
                all_probs.append(probs.detach())

        if grad:
            return total_loss / max(B, 1), None
        else:
            pred = torch.cat(all_probs, dim=0) if all_probs else torch.zeros_like(knn[:0])
            metric = full_metrics(pred, true.index_select(0, poi_rows), inv.index_select(0, poi_rows))
            return 0.0, metric

    # Train (fixed epochs) and pick best-by-val-KL
    best_val, best_state = float("inf"), None
    for ep in range(1, cfg.epochs + 1):
        train_loss, _ = run_epoch("train", grad=True)
        _, val_m = run_epoch("val", grad=False)
        if val_m["kl"] + 1e-6 < best_val:
            best_val, best_state = val_m["kl"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep == 1) or (ep % 10 == 0) or (ep == cfg.epochs):
            print(f"Ep{ep:03d} | train_KL {train_loss:.6f} | "
                  f"val: KL {val_m['kl']:.6f} MAE {val_m['mae']:.6f} "
                  f"T1 {val_m['top1']:.3f} MRR {val_m['mrr']:.3f} NDCG {val_m['ndcg']:.3f} R2 {val_m['r2']:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # Final eval on test (or val if test missing)
    _, test_m = run_epoch("test", grad=False)
    print("Final TEST | KL {:.6f}  MAE {:.6f}  Top-1 {:.3f}  MRR {:.3f}  NDCG {:.3f}  R2 {:.3f}".format(
        test_m["kl"], test_m["mae"], test_m["top1"], test_m["mrr"], test_m["ndcg"], test_m["r2"]))

    # Save predictions for ALL POIs on EVAL-K (so you can compare apples-to-apples)
    with torch.no_grad():
        B = int(data['poi'].num_nodes); bs = cfg.batch_size
        rows = torch.arange(B, device=cfg.device)
        preds = torch.empty((B, cfg.k_eval), device=cfg.device)
        for start in range(0, B, bs):
            end = min(B, start + bs)
            idx = rows[start:end]
            k_idx = knn_eval.index_select(0, idx)
            inv_m = inv_eval.index_select(0, idx)
            d_sub = dist_eval.index_select(0, idx) if cfg.use_distance else None
            probs = model.predict_probs(data['poi'].x, data['cbg'].x, k_idx, inv_m, d_sub)
            preds[start:end] = probs

    # Write CSV: poi_node_id, rank_in_knn, cbg_node_id, pred_prob
    out_pred = []
    knn_np = knn_eval.detach().cpu().numpy()
    pred_np= preds.detach().cpu().numpy()
    N, K = knn_np.shape
    for i in range(N):
        for j in range(K):
            c = int(knn_np[i, j])
            if c >= 0:
                out_pred.append({"poi_node_id": i, "rank_in_knn": j+1,
                                 "cbg_node_id": c, "pred_prob": float(pred_np[i, j])})
    df_pred = pd.DataFrame(out_pred)
    pred_path = os.path.join(cfg.out_dir, "mlp_preds.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"Saved predictions -> {pred_path}")

    # Save aggregate metrics
    agg_path = os.path.join(cfg.out_dir, "mlp_agg_metrics.json")
    with open(agg_path, "w") as f:
        json.dump(test_m, f, indent=2)
    print(f"Saved aggregate metrics -> {agg_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_path", required=True)
    ap.add_argument("--out_dir", default="/mnt/data/mlp_baseline_outputs")
    ap.add_argument("--k_train", type=int, default=40)
    ap.add_argument("--k_eval",  type=int, default=40)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--poi_hidden", type=int, default=128)
    ap.add_argument("--cbg_hidden", type=int, default=256)
    ap.add_argument("--pair_hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--use_distance", dest="use_distance", action="store_true")
    ap.add_argument("--no_use_distance", dest="use_distance", action="store_false")
    ap.set_defaults(use_distance=True)

    ap.add_argument("--feature_variant", default="base+dwell",
                    choices=["base","base+text","base+dwell","base+text+dwell"])
    ap.add_argument("--poi_features_json", default=None)
    ap.add_argument("--cbg_features_json", default=None)

    args = ap.parse_args()
    cfg = Config(graph_path=args.graph_path,
                 out_dir=args.out_dir,
                 k_train=args.k_train, k_eval=args.k_eval,
                 epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                 batch_size=args.batch_size, seed=args.seed, device=args.device,
                 poi_hidden=args.poi_hidden, cbg_hidden=args.cbg_hidden,
                 pair_hidden=args.pair_hidden, dropout=args.dropout,
                 use_distance=args.use_distance,
                 feature_variant=args.feature_variant,
                 poi_features_json=args.poi_features_json,
                 cbg_features_json=args.cbg_features_json)
    run(cfg)

if __name__ == "__main__":
    main()




# python /home/lp43319/projects/GNN/visitgnn/mlp_baseline.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1_poi_poi/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --out_dir /home/lp43319/projects/GNN/visitgnn/baseline \
#   --k_train 50 --k_eval 50 \
#   --epochs 800 --lr 1e-3 --batch_size 2048 \
#   --feature_variant base+dwell \
#   --poi_hidden 128 --cbg_hidden 256 --pair_hidden 64 \
#   --use_distance \
#   --seed 42
