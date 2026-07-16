#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna HPO for VisitHeteroGNN (Plan A: unified search space + optional fixed K)

- 丢弃所有 visit 边（仅用于构造监督，不参与 message passing）
- ('poi','knn','cbg') 及其反向：保留并传 edge_attr
- ('cbg','adjacent','cbg')：保留拓扑（无属性）
- 若 --use_poi_poi：纳入 POI↔POI（geo_knn/time_sim/brand）及其 edge_attr
- 训练/验证一律通过 build_edge_index_for_train(...) 构建，确保 edge_attr 对齐、自环已去
- 统一且**相同**的超参搜索空间（开启/关闭 POI–POI 均一致）
- 可通过 --fixed_k 固定 K（推荐固定，确保 KL 指标可比）
"""

import argparse as _ap
import json as _json
import os as _os
from functools import lru_cache as _lru_cache
import inspect as _inspect

import numpy as _np
import optuna as _optuna
import torch as _torch

from train import (
    VisitHeteroGNN,
    build_targets_from_knn_candidates,
    build_edge_index_for_train,
    masked_metrics,
    masked_kl_loss,
    set_seed,
)

# -------------------------------
# Utils
# -------------------------------

@_lru_cache(maxsize=1)
def _load_graph_raw(path: str):
    print(f"[Cache] loading graph from {path}")
    return _torch.load(path, weights_only=False, map_location="cpu")

def _as_heterodata(obj):
    if isinstance(obj, dict) and "graph_data" in obj:
        return obj["graph_data"]
    return obj

def _split_pois(num_poi: int, seed: int = 42):
    rng = _np.random.default_rng(seed)
    idx = _np.arange(num_poi); rng.shuffle(idx)
    n_train = int(0.70 * num_poi)
    n_val   = int(0.15 * num_poi)
    return (
        _torch.tensor(idx[:n_train], dtype=_torch.long),
        _torch.tensor(idx[n_train:n_train + n_val], dtype=_torch.long),
        _torch.tensor(idx[n_train + n_val:], dtype=_torch.long),
    )

def _get_splits(data, seed: int):
    def _pick(k):
        if hasattr(data, k): return getattr(data, k)
        if isinstance(data, dict) and (k in data): return data[k]
        return None
    tr = _pick("train_idx"); va = _pick("val_idx"); te = _pick("test_idx")
    if tr is not None and va is not None:
        return tr, va, te
    return _split_pois(int(data["poi"].num_nodes), seed=seed)

def _is_knn_rel(et):
    s, rel, d = et
    return (s == 'poi' and rel == 'knn' and d == 'cbg') or \
           (s == 'cbg' and rel in ('rev_knn', 'knn__rev') and d == 'poi')

def _is_visit_rel(et):
    return et[1] in {'visit', 'rev_visit', 'visit__rev'}

def _is_poi_poi_rel(et):
    s, rel, d = et
    return (s == 'poi') and (d == 'poi') and (rel in {'geo_knn', 'time_sim', 'brand'})

def _keep_relations(edge_types, use_poi_poi: bool):
    present = set(edge_types)
    keep = {('poi','belong','cbg'),
            ('poi','knn','cbg'),
            ('cbg','rev_knn','poi'),
            ('cbg','rev_belong','poi'),
            ('cbg','adjacent','cbg')}
    if use_poi_poi:
        keep |= {et for et in present if _is_poi_poi_rel(et)}
    keep = {et for et in keep if et in present and not _is_visit_rel(et)}
    return keep

def _supports_kw(fn, name: str) -> bool:
    try:
        return name in _inspect.signature(fn).parameters
    except Exception:
        return False

def _build_model(data, dims, use_poi_poi: bool, device):
    """仅在支持时传 use_poi_poi/include_visit_edges；不传 fallback。"""
    poi_in_dim = int(data['poi'].x.size(1))
    cbg_in_dim = int(data['cbg'].x.size(1))
    d_cbg, d_poi, d_hidden, dropout = dims

    ctor = VisitHeteroGNN.__init__
    kws = {}
    if _supports_kw(ctor, 'poi_in_dim'):   kws['poi_in_dim'] = poi_in_dim
    if _supports_kw(ctor, 'cbg_in_dim'):   kws['cbg_in_dim'] = cbg_in_dim
    if _supports_kw(ctor, 'd_cbg'):        kws['d_cbg'] = d_cbg
    if _supports_kw(ctor, 'd_poi'):        kws['d_poi'] = d_poi
    if _supports_kw(ctor, 'd_hidden'):     kws['d_hidden'] = d_hidden
    if _supports_kw(ctor, 'dropout'):      kws['dropout'] = dropout
    if _supports_kw(ctor, 'include_visit_edges'):
        kws['include_visit_edges'] = False
    if _supports_kw(ctor, 'use_poi_poi'):
        kws['use_poi_poi'] = use_poi_poi

    return VisitHeteroGNN(**kws).to(device)

def _filter_idx_attr(idx_d, attr_d, keep):
    # 仅保留需要的关系；且只给 KNN/POI-POI 传 edge_attr
    def _is_poi_poi(et): return _is_poi_poi_rel(et)
    def _is_knn(et):     return _is_knn_rel(et)
    idx_d = {et: ei for et, ei in idx_d.items() if et in keep}
    attr_d = {et: ea for et, ea in attr_d.items()
              if et in keep and ((_is_knn(et) or _is_poi_poi(et)) and (ea is not None))}
    return idx_d, attr_d

# -------------------------------
# Objective
# -------------------------------

def _objective(trial: _optuna.Trial, args) -> float:
    try:
        set_seed(args.seed)
        device = _torch.device(args.device)

        # 统一搜索空间（与是否 use_poi_poi 无关）
        lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout      = trial.suggest_float("dropout", 0.05, 0.50)
        d_cbg        = trial.suggest_categorical("d_cbg", [128, 256])
        d_poi        = trial.suggest_categorical("d_poi", [64, 128, 256])
        d_hidden     = trial.suggest_categorical("d_hidden", [64, 128, 256])
        k_final      = args.fixed_k if args.fixed_k is not None else trial.suggest_int("k", 50, 50)

        # 数据与 targets
        bundle = _load_graph_raw(args.graph_path)
        data = _as_heterodata(bundle).to(device)
        knn_idx, true_probs, invalid = build_targets_from_knn_candidates(data, k_final, device)

        train_idx, val_idx, _ = _get_splits(data, args.seed)
        train_idx = train_idx.to(device); val_idx = val_idx.to(device)

        # 模型/优化器
        model = _build_model(
            data,
            dims=(d_cbg, d_poi, d_hidden, dropout),
            use_poi_poi=args.use_poi_poi,
            device=device
        )
        optim  = _torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # ---------- 训练/验证都用 build_edge_index_for_train ----------
        edge_train_idx, edge_train_attr_all = build_edge_index_for_train(data, train_idx)
        all_poi = _torch.arange(int(data['poi'].num_nodes), device=device)
        edge_full_idx_all, edge_full_attr_all = build_edge_index_for_train(data, all_poi)

        KEEP = _keep_relations(getattr(data, 'edge_types', []), use_poi_poi=args.use_poi_poi)
        edge_train_idx, edge_train_attr = _filter_idx_attr(edge_train_idx, edge_train_attr_all, KEEP)
        edge_full_idx,  edge_full_attr  = _filter_idx_attr(edge_full_idx_all, edge_full_attr_all, KEEP)

        # ---- 早停训练（以验证 KL 为准）----
        best_val, no_imp = float("inf"), 0
        for ep in range(1, args.max_epochs + 1):
            model.train(); optim.zero_grad(set_to_none=True)
            z = model(data.x_dict, edge_train_idx, edge_attr_dict=edge_train_attr)
            p = model.predict_probs(z, knn_idx, invalid)
            loss = masked_kl_loss(
                p.index_select(0, train_idx),
                true_probs.index_select(0, train_idx),
                invalid.index_select(0, train_idx)
            )
            loss.backward(); optim.step()

            model.eval()
            with _torch.no_grad():
                zv = model(data.x_dict, edge_full_idx, edge_attr_dict=edge_full_attr)
                pv = model.predict_probs(zv, knn_idx, invalid)
                metr_va = masked_metrics(
                    pv.index_select(0, val_idx),
                    true_probs.index_select(0, val_idx),
                    invalid.index_select(0, val_idx)
                )
            val_kl = metr_va["kl"]

            trial.report(val_kl, ep)
            if trial.should_prune():
                raise _optuna.TrialPruned()

            if val_kl < best_val - 1e-6:
                best_val, no_imp = val_kl, 0
            else:
                no_imp += 1
                if no_imp >= args.patience:
                    break

        return best_val

    except _torch.cuda.OutOfMemoryError:
        _torch.cuda.empty_cache()
        raise _optuna.TrialPruned()
    finally:
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

# -------------------------------
# CLI
# -------------------------------

def main():
    ap = _ap.ArgumentParser()
    ap.add_argument("--graph_path", required=True)
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--device", default="cuda" if _torch.cuda.is_available() else "cpu")

    # 统一搜索空间；提供固定 K（强烈建议在公平对比时设置）
    ap.add_argument("--fixed_k", type=int, default=50,
                    help="固定 top-K 候选（默认 30）。为公平对比，建议显式设置一个固定值。")

    # 仅 POI–POI 开关（无 fallback）
    ap.add_argument("--use_poi_poi", dest="use_poi_poi", action="store_true")
    ap.add_argument("--no_poi_poi",  dest="use_poi_poi", action="store_false")
    ap.set_defaults(use_poi_poi=True)

    args = ap.parse_args()

    _os.makedirs(args.output_dir, exist_ok=True)
    _optuna.logging.set_verbosity(_optuna.logging.INFO)
    study = _optuna.create_study(
        direction="minimize",
        sampler=_optuna.samplers.TPESampler(seed=args.seed)
    )
    study.optimize(lambda t: _objective(t, args), n_trials=args.trials)

    print(f"Best KL    : {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    out_json = _os.path.join(args.output_dir, "best_params.json")
    with open(out_json, "w") as f:
        _json.dump(study.best_params, f, indent=2)
    print(f"✓ Saved best parameters → {out_json}")

if __name__ == "__main__":
    main()

# python /home/lp43319/projects/GNN/visitgnn/hyper.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/athens_w1/athens_w1_hetero_with_text_and_poi_edges_split.pt \
#   --output_dir /home/lp43319/projects/GNN/visitgnn/output/athens_w1/output/hyper \
#   --trials 200 --max_epochs 180 --patience 20 --device cuda:0 \
#   --fixed_k 10 \
#   --no_poi_poi


# # B：关闭 POI–POI（其他完全相同）
# python /home/lp43319/projects/GNN/visitgnn/hyper.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --output_dir /home/lp43319/projects/GNN/visitgnn/output/hyper \
#   --trials 200 --max_epochs 180 --patience 20 --device cuda:0 \
#   --fixed_k 50 \
#   --no_poi_poi








