#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for VisitHeteroGNN (compatible with your latest train.py)

Key points
- Build edges like train: drop self-loops for homogeneous edges; align/pad edge_attr.
- Keep/Drop POI–POI at inference (default = follow checkpoint; can override by flags).
- Pass edge_attr only to relations that need it (KNN + POI–POI).
- Derive d_hidden (for pred_mlp) and POI–POI edge_dim from checkpoint to avoid shape mismatch.
- Build convs first (with correct edge_dim), then load a SHAPE-FILTERED state_dict (skip mismatched buffers like *_mean/_std).
- Emit per-(poi, cbg, rank) predictions, and enrich with mapping CSV.
"""

from __future__ import annotations
import argparse
import os
import json
import shutil
import inspect
import math
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Import your model & helper from train.py
# from train1001 import VisitHeteroGNN, build_targets_from_knn_candidates
from train_pp_optimized import VisitHeteroGNN, build_targets_from_knn_candidates

# ------------------------------
# I/O helpers
# ------------------------------
def enrich_with_mapping(df: pd.DataFrame, mapping_csv: str) -> pd.DataFrame:
    map_df = pd.read_csv(mapping_csv).rename(columns={"node_id": "poi_node_id", "rank": "rank_in_knn"})
    for col in ["poi_node_id", "cbg_node_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        if col in map_df.columns:
            map_df[col] = pd.to_numeric(map_df[col], errors="coerce").astype("Int64")
    if "CBG_ID" in map_df.columns:
        map_df["CBG_ID"] = map_df["CBG_ID"].astype(str)
    if "placekey" in map_df.columns:
        map_df["placekey"] = map_df["placekey"].astype(str)

    if set(["poi_node_id", "cbg_node_id", "rank_in_knn"]).issubset(map_df.columns):
        map_df = (map_df.sort_values("rank_in_knn")
                        .drop_duplicates(["poi_node_id", "cbg_node_id"], keep="first"))
    keep = [c for c in ["poi_node_id", "cbg_node_id", "CBG_ID", "placekey"] if c in map_df.columns]
    return df.merge(map_df[keep].drop_duplicates(), on=["poi_node_id", "cbg_node_id"], how="left")


def _expected_input_dims_from_ckpt(state_dict: dict) -> tuple[int, int]:
    try:
        exp_poi = int(state_dict['poi_mlp.0.weight'].shape[1])
        exp_cbg = int(state_dict['cbg_proj.weight'].shape[1])
        return exp_poi, exp_cbg
    except KeyError as e:
        raise RuntimeError(f"Checkpoint missing keys to infer input dims: {e}")


def _adapt_feature_dim(x: torch.Tensor, expected: int) -> torch.Tensor:
    cur = x.size(1)
    if cur == expected:
        return x
    if cur < expected:
        pad = torch.zeros(x.size(0), expected - cur, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)
    return x[:, :expected]


# ------------------------------
# Edge helpers (build like train)
# ------------------------------
POI_POI_SET = {"geo_knn", "time_sim", "brand"}

def _base_rel(rel: str) -> str:
    if rel.startswith("rev_"): rel = rel[4:]
    if rel.endswith("__rev"):  rel = rel[:-5]
    return rel

def _is_visit(et):   return et[1] in {"visit","rev_visit","visit__rev"}
def _is_poi_poi(et): return et[0]=="poi" and et[2]=="poi"
def _is_knn(et):     return _base_rel(et[1])=="knn" and {et[0],et[2]}=={"poi","cbg"}

def build_edges_and_attrs_for_infer(data: HeteroData, keep_poi_poi: bool):
    """
    Train-consistent edge building:
      - drop visit; optionally drop POI–POI;
      - drop self-loops for homogeneous edges;
      - align edge_attr to filtered edges (pad/trim if needed);
      - return edge_attr only for relations that need it (KNN + POI–POI).
    """
    edge_index_dict, edge_attr_aligned = {}, {}

    for et in data.edge_types:
        if _is_visit(et):
            continue
        if _is_poi_poi(et) and (not keep_poi_poi):
            continue

        s, rel, d = et
        ei = data[et].edge_index                      # [2, E]
        ea = getattr(data[et], "edge_attr", None)     # [E,*] or [E_noloop,*] or None
        E = ei.size(1)

        # drop self-loops for homogeneous edges
        non_loop = torch.ones(E, dtype=torch.bool, device=ei.device)
        if s == d:
            non_loop = (ei[0] != ei[1])

        keep = non_loop
        ei_new = ei[:, keep]
        edge_index_dict[et] = ei_new

        if ea is None:
            continue

        if ea.size(0) == E:
            ea_sel = ea[keep]
        elif ea.size(0) == int(non_loop.sum().item()):
            keep_in_nl = keep[non_loop]
            ea_sel = ea[keep_in_nl]
        else:
            # repair to match expected target length then select
            target_len = int(non_loop.sum().item()) if (s == d) else E
            if ea.size(0) < target_len:
                pad_shape = (target_len - ea.size(0),) + tuple(ea.size()[1:])
                pad = torch.zeros(pad_shape, dtype=ea.dtype, device=ea.device)
                ea_fix = torch.cat([ea, pad], dim=0)
            else:
                ea_fix = ea[:target_len]
            if s == d:
                keep_in_nl = keep[non_loop]
                ea_sel = ea_fix[keep_in_nl]
            else:
                ea_sel = ea_fix[keep]

        edge_attr_aligned[et] = ea_sel

    # only relations that need edge_attr (KNN + POI–POI)
    edge_attr_needed = {}
    for et, ea in edge_attr_aligned.items():
        s, rel, d = et
        base = _base_rel(rel)
        needs_attr = (_is_knn(et)) or (s=="poi" and d=="poi" and base in POI_POI_SET)
        if needs_attr:
            edge_attr_needed[et] = ea

    return edge_index_dict, edge_attr_needed


# ------------------------------
# Detect ckpt architecture/meta
# ------------------------------
def ckpt_has_poi_poi(state: dict) -> bool:
    return any(k.startswith('poi_poi_conv.') for k in state.keys())

def ckpt_hidden_dim(state: dict, fallback: int) -> int:
    # pred_mlp.0.bias has length = d_hidden
    key = 'pred_mlp.0.bias'
    if key in state:
        return int(state[key].numel())
    return fallback

def ckpt_edge_dim_for_rel(state: dict, rel_name: str, default_dim: int | None) -> int | None:
    """
    Try to discover edge_dim used by GATv2Conv for a specific POI-POI relation.
    Look for any key that contains 'poi_poi_conv.convs' + rel_name + 'lin_edge.weight'.
    """
    for k, v in state.items():
        if ("poi_poi_conv.convs" in k) and ("lin_edge.weight" in k) and (rel_name in k):
            try:
                return int(v.shape[1])
            except Exception:
                pass
    return default_dim


# ------------------------------
# State-dict filtering (shape-safe)
# ------------------------------
def build_shape_compatible_state(model: torch.nn.Module, full_state: dict):
    """
    Return a dict that only contains keys present in model.state_dict()
    AND whose tensor shapes exactly match. Everything else is skipped.
    """
    cur = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in full_state.items():
        if k not in cur:
            continue
        if tuple(cur[k].shape) != tuple(v.shape):
            skipped.append((k, tuple(v.shape), tuple(cur[k].shape)))
            continue
        loadable[k] = v
    missing_in_ckpt = [k for k in cur.keys() if k not in full_state]
    unexpected_in_ckpt = [k for k in full_state.keys() if k not in cur]
    return loadable, skipped, missing_in_ckpt, unexpected_in_ckpt


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_path",   required=True)
    ap.add_argument("--ckpt_path",    required=True)   # model.state_dict() .pt
    ap.add_argument("--params_json",  required=True)   # hyperparams (k, dims)
    ap.add_argument("--mapping_csv",  required=True)
    ap.add_argument("--output_dir",   required=True)
    ap.add_argument("--out_csv",      default=None)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--force_no_poi_poi",   action="store_true")
    g.add_argument("--force_with_poi_poi", action="store_true")
    ap.add_argument("--strict_schema", action="store_true")
    ap.add_argument("--poi_poi_modes", default="geo_knn,time_sim,brand",
                    help="comma-separated subset of {geo_knn,time_sim,brand}. "
                         "Use 'geo_knn,time_sim' for models trained WITHOUT brand "
                         "(avoids building a randomly-initialized brand conv at inference).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- hyperparams from json ----
    try:
        with open(args.params_json, "r", encoding="utf-8") as f:
            best = json.load(f)
    except UnicodeDecodeError:
        raise RuntimeError(f"[params_json] {args.params_json} 不是 UTF-8 文本文件（别把 .pt 当 json 传）")

    K        = int(best.get("k", 50))
    d_hidden = int(best.get("d_hidden", 64))
    d_cbg    = int(best.get("d_cbg", 256))
    d_poi    = int(best.get("d_poi", 128))
    dropout  = float(best.get("dropout", 0.10))

    # ---- load graph & checkpoint ----
    data: HeteroData = torch.load(args.graph_path, weights_only=False).to(device)
    full_state = torch.load(args.ckpt_path, map_location="cpu")
    exp_poi_in, exp_cbg_in = _expected_input_dims_from_ckpt(full_state)

    # override d_hidden by ckpt to match pred_mlp
    d_hidden_ckpt = ckpt_hidden_dim(full_state, d_hidden)
    if d_hidden_ckpt != d_hidden:
        print(f"[warn] override d_hidden: json={d_hidden} -> ckpt={d_hidden_ckpt}")
        d_hidden = d_hidden_ckpt

    g_poi_in = int(data['poi'].x.size(1))
    g_cbg_in = int(data['cbg'].x.size(1))
    print(f"[info] graph dims poi/cbg = {g_poi_in}/{g_cbg_in} ; ckpt expects = {exp_poi_in}/{exp_cbg_in}")

    if args.strict_schema:
        assert g_poi_in == exp_poi_in and g_cbg_in == exp_cbg_in, \
            "Feature dims mismatch. Rebuild graph to match training schema."
    else:
        if g_poi_in != exp_poi_in:
            print(f"[warn] Adapting POI features: {g_poi_in} → {exp_poi_in}")
            data['poi'].x = _adapt_feature_dim(data['poi'].x, exp_poi_in)
        if g_cbg_in != exp_cbg_in:
            print(f"[warn] Adapting CBG features: {g_cbg_in} → {exp_cbg_in}")
            data['cbg'].x = _adapt_feature_dim(data['cbg'].x, exp_cbg_in)

    # ---- whether to keep POI–POI edges ----
    ckpt_poi_poi = ckpt_has_poi_poi(full_state)
    keep_poi_poi = ckpt_poi_poi
    if args.force_no_poi_poi:   keep_poi_poi = False
    if args.force_with_poi_poi: keep_poi_poi = True
    print(f"[info] ckpt has POI-POI: {ckpt_poi_poi} ; keep_poi_poi at inference: {keep_poi_poi}")

    # ---- build edges & edge_attr (train-consistent) ----
    edge_index_dict, edge_attr_dict = build_edges_and_attrs_for_infer(data, keep_poi_poi=keep_poi_poi)
    print(f"[info] edge types used in forward ({len(edge_index_dict)}): {sorted(map(str, edge_index_dict.keys()))}")

    # ---- prepare constructor kwargs (only pass supported optional args) ----
    base_kwargs = dict(
        poi_in_dim=exp_poi_in,
        cbg_in_dim=exp_cbg_in,
        d_cbg=d_cbg,
        d_poi=d_poi,
        d_hidden=d_hidden,
        dropout=dropout,
        include_visit_edges=False,
    )

    # default optional args matching your latest train.py
    opt_default = dict(
        use_poi_poi=keep_poi_poi,
        poi_poi_modes=tuple(s.strip() for s in args.poi_poi_modes.split(",") if s.strip()),
        aggr='mean',
        use_edge_norm=True,
        edge_temp={'geo_knn':1.0,'time_sim':1.0,'brand':1.5},
        rel_scale={'geo_knn':1.0,'time_sim':1.0,'brand':0.5},
        edge_mlp_dim=4,
        use_fallback=True,
    )

    # Override edge_mlp_dim by ckpt POI–POI edge_dim if detectable
    if keep_poi_poi:
        edims = []
        for rel in ('geo_knn','time_sim','brand'):
            ed = ckpt_edge_dim_for_rel(full_state, rel, None)
            if ed is not None:
                edims.append(ed)
        if edims:
            edim_ckpt = int(edims[0])
            if opt_default['edge_mlp_dim'] != edim_ckpt:
                print(f"[warn] override edge_mlp_dim: default={opt_default['edge_mlp_dim']} -> ckpt={edim_ckpt}")
                opt_default['edge_mlp_dim'] = edim_ckpt

    # only pass options supported by current class signature
    sig = inspect.signature(VisitHeteroGNN.__init__)
    opt_supported = {k: v for k, v in opt_default.items() if k in sig.parameters}

    model = VisitHeteroGNN(**base_kwargs, **opt_supported).to(device)

    # ---- build convs BEFORE loading weights ----
    if hasattr(model, "_build_poi_poi_conv") and keep_poi_poi:
        model._build_poi_poi_conv(edge_index_dict)
    if hasattr(model, "_build_cross_conv"):
        model._build_cross_conv(edge_index_dict)

    # ---- load SHAPE-COMPATIBLE state_dict (skip mismatched buffers like *_mean/_std) ----
    loadable, skipped, missing, unexpected = build_shape_compatible_state(model, full_state)
    if unexpected:
        print(f"[info] ignore unexpected keys (up to 10): {unexpected[:10]}")
    if missing:
        print(f"[info] missing keys for this graph/model (up to 10): {missing[:10]}")
    if skipped:
        shown = "\n".join([f"  - {k}: ckpt{shp_ckpt} != model{shp_model}" for k, shp_ckpt, shp_model in skipped[:10]])
        print(f"[info] skipped {len(skipped)} mismatched tensors due to shape diff.\n{shown}")
    model.load_state_dict(loadable, strict=False)
    model.eval()

    # ---- KNN candidates & ground truth ----
    knn_idx, true_probs, invalid = build_targets_from_knn_candidates(data, K, device)

    # ---- forward ----
    with torch.no_grad():
        z = model(
            data.x_dict,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict if edge_attr_dict else None
        )
        pred = model.predict_probs(z, knn_idx, invalid).cpu().numpy()
        true = true_probs.cpu().numpy()
        knn_np = knn_idx.cpu().numpy()

    # ---- write rows ----
    rows = []
    N, Kc = knn_np.shape
    for i in range(N):
        for j in range(Kc):
            c = int(knn_np[i, j])
            if c < 0:
                continue
            rows.append({
                "poi_node_id": i,
                "rank_in_knn": j + 1,
                "cbg_node_id": c,
                "ground_truth": float(true[i, j]),
                "pred_prob":   float(pred[i, j]),
            })
    df = pd.DataFrame(rows)
    df = enrich_with_mapping(df, args.mapping_csv)

    out_csv = args.out_csv or os.path.join(
        args.output_dir,
        f"{os.path.splitext(os.path.basename(args.graph_path))[0]}_preds.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"✓ predictions → {out_csv}")

    dst_map = os.path.join(args.output_dir, os.path.basename(args.mapping_csv))
    if os.path.abspath(dst_map) != os.path.abspath(args.mapping_csv):
        shutil.copy(args.mapping_csv, dst_map)
        print(f"✓ copied mapping → {dst_map}")


if __name__ == "__main__":
    main()


# python /home/lp43319/projects/GNN/visitgnn/inferencecopy.py \
#     --graph_path   /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#     --ckpt_path    /home/lp43319/projects/GNN/visitgnn/output/Train_pp_optimized/best_visit_gnn.pt \
#     --params_json  /home/lp43319/projects/GNN/visitgnn/output/hyper/best_params.json \
#     --mapping_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_to_cbg_mapping.csv \
#     --output_dir   /home/lp43319/projects/GNN/visitgnn/output/Train_pp_optimized/prediction \
#     --device       cuda:0


# python /home/lp43319/projects/GNN/visitgnn/inferencecopy.py \
#     --graph_path   /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#     --ckpt_path    /home/lp43319/projects/GNN/visitgnn/output/Train_pp_good/best_visit_gnn.pt \
#     --params_json  /home/lp43319/projects/GNN/visitgnn/output/hyper/best_params.json \
#     --mapping_csv  /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_poi_to_cbg_mapping.csv \
#     --output_dir   /home/lp43319/projects/GNN/visitgnn/output/Train_pp_good/prediction \
#     --device       cuda:0


