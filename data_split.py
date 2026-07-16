#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create POI train/val/test splits and store them inside a new .pt graph file.
Works with the HeteroData produced by build_heterograph.py.
"""

import argparse
import os
from typing import Tuple, List

import numpy as np
import torch


def split_poi_indices(
    num_poi: int, ratios=(0.7, 0.15, 0.15), seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic split over POI nodes. Returns CPU int64 tensors."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    rng = np.random.default_rng(seed)
    order = np.arange(num_poi)
    rng.shuffle(order)
    n_train = int(ratios[0] * num_poi)
    n_val   = int(ratios[1] * num_poi)
    train_idx = torch.as_tensor(order[:n_train], dtype=torch.long)
    val_idx   = torch.as_tensor(order[n_train:n_train + n_val], dtype=torch.long)
    test_idx  = torch.as_tensor(order[n_train + n_val:], dtype=torch.long)
    return train_idx, val_idx, test_idx


def infer_num_poi_from_edges(data) -> int:
    """Infer POI count from any edge where POI appears as src or dst."""
    candidates: List[int] = []
    for et in getattr(data, "edge_types", []):
        src_t, _, dst_t = et
        ei = data[et].edge_index
        if ei.numel() == 0:
            continue
        if src_t == "poi":
            candidates.append(int(ei[0].max().item()) + 1)
        if dst_t == "poi":
            candidates.append(int(ei[1].max().item()) + 1)
    return max(candidates) if candidates else 0


def main():
    ap = argparse.ArgumentParser(
        description="Create POI train/val/test splits and store inside the .pt graph"
    )
    ap.add_argument("--graph_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph not found: {args.graph_path}")

    data = torch.load(args.graph_path, weights_only=False)

    # ---- robust POI existence & count ----
    try:
        poi_store = data["poi"]  # don't use `"poi" in data"`; it's unreliable in PyG
    except Exception:
        raise RuntimeError("Graph has no 'poi' node type.")

    # priority: x.shape[0] -> num_nodes -> infer from edges
    num_poi = 0
    x = getattr(poi_store, "x", None)
    if x is not None:
        num_poi = int(x.size(0))
    if num_poi == 0:
        num_poi = int(getattr(poi_store, "num_nodes", 0) or 0)
    if num_poi == 0:
        num_poi = infer_num_poi_from_edges(data)

    if num_poi <= 0:
        raise RuntimeError("Cannot infer POI node count (no x/num_nodes/edges).")

    print(f"[split] num_poi = {num_poi}")

    # ---- split and save ----
    tr, va, te = split_poi_indices(num_poi, seed=args.seed)
    data["train_idx"], data["val_idx"], data["test_idx"] = tr, va, te

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(data, args.out_path)
    print(f"✓ Saved graph with splits → {args.out_path}")
    print(f"  train={len(tr)}  val={len(va)}  test={len(te)}")


if __name__ == "__main__":
    main()




# python /home/lp43319/projects/GNN/visitgnn/data_split.py \
#   --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges.pt \
#   --out_path   /home/lp43319/projects/GNN/visitgnn/output/fulton_w1/fulton_w1_hetero_with_text_and_poi_edges_split.pt \
#   --seed 42


