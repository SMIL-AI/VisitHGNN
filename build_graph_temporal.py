
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POI ↔ CBG pipeline with stable POI/CBG feature schemas (+ optional rich POI embeddings)
+ Optional POI↔POI edges from PLGOT modules (geo KNN + time-sim + brand + co-occ).

--------------------------------------------------------------------------------------
- Load POI/CBG tables and CBG shapefile
- Enrich POIs: opening-hours features, category encodings
- Optional: Attach extra POI embeddings such as BERT text (npy/csv) and dwell vectors (npy)
- Derive clean POI features with a stable schema (column order locked to schema files)
- Build GeoDataFrames (UTM), distance-KNN (POI→CBG)
- Assemble PyG HeteroData with edges:
    ('poi','belong','cbg'), ('poi','knn','cbg'),
    ('cbg','adjacent','cbg'), ('cbg','visit','poi')
  and optional POI↔POI edges (if provided):
    ('poi','geo_knn','poi'), ('poi','time_sim','poi'),
    ('poi','brand','poi')
- Save artifacts and graph .pt

This version:
- REMOVED legacy reuse flags and logic:
  --reuse_poi_sample_csv
  --reuse_mapping_csv
- Added explicit print summaries for relations/edge_attr.

Example:
python Build_Heterograph.py \
  /path/to/Fulton_POI_week1_with_macro.csv \
  /path/to/Fulton_cbg.csv \
  /path/to/tl_2019_13_bg.shp \
  /path/to/output_dir \
  --run_name fulton_w1 \
  --k 544 --sample_size 8000 \
  --poi_text_csv /path/to/mydataset_geo.parquet \
  --plgot_poi_csv /path/to/mydataset.csv \
  --plgot_dwell_npy /path/to/dwell_feat.npy \
  --plgot_knn_npz  /path/to/knn_edges.npz \
  --plgot_time_npz /path/to/time_similarity_edges.npz \
  --plgot_brand_npz /path/to/brand_week_edges.npz \
  --l2_norm_text \
  --fips_prefix 13121 --min_visits 10 \
  --out_graph_name graph_text_and_poi_edges.pt
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


# ------------------------------
# Logging
# ------------------------------

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------
# Utils
# ------------------------------

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def to_hours_dict(val) -> Dict[str, list]:
    if val is None:
        return {}
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def day_hours(intervals: Iterable[Sequence[str]]) -> float:
    total = timedelta()
    for iv in intervals or []:
        if not isinstance(iv, (list, tuple)) or len(iv) != 2:
            continue
        start, end = iv
        try:
            s = datetime.strptime(str(start), "%H:%M")
            e = datetime.strptime(str(end), "%H:%M")
            if e <= s:
                e += timedelta(days=1)
            total += (e - s)
        except Exception:
            continue
    return round(total.total_seconds() / 3600.0, 2)


def ensure_cbg12(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        s = str(int(float(x)))
        return s.zfill(12)
    except Exception:
        s = str(x).strip()
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            return digits.zfill(12)[:12]
        return None


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------
# Feature engineering (POI)
# ------------------------------

POI_FEATURE_ALLOWLIST = [
    # geometry / intensity / dwell
    "wkt_area_sq_meters", "raw_visit_counts", "raw_visitor_counts",
    "distance_from_home", "median_dwell",
    # normalized stats
    "normalized_visits_by_state_scaling",
    "normalized_visits_by_region_naics_visits",
    "normalized_visits_by_region_naics_visitors",
    "normalized_visits_by_total_visits",
    "normalized_visits_by_total_visitors",
    # categories (prepared below)
    "top_category_enc", "macro_enc",
    # opening hours
    "open_mon_hours", "open_tue_hours", "open_wed_hours", "open_thu_hours",
    "open_fri_hours", "open_sat_hours", "open_sun_hours", "open_days_in_week",
]


def compute_open_hours_features(pois: pd.DataFrame) -> pd.DataFrame:
    pois = pois.copy()
    if "open_hours" not in pois.columns:
        for d in DAYS:
            pois[f"open_{d.lower()}_hours"] = 0.0
        pois["open_days_in_week"] = 0
        return pois

    pois["hours_dict"] = pois["open_hours"].apply(to_hours_dict)
    for d in DAYS:
        col = f"open_{d.lower()}_hours"
        pois[col] = pois["hours_dict"].apply(lambda dd: day_hours(dd.get(d, [])))
    hour_cols = [f"open_{d.lower()}_hours" for d in DAYS]
    pois["open_days_in_week"] = pois[hour_cols].gt(0).sum(axis=1)
    return pois


def encode_categories(pois: pd.DataFrame) -> pd.DataFrame:
    pois = pois.copy()
    if "top_category" in pois.columns:
        pois["top_category_enc"] = pois["top_category"].astype("category").cat.codes
    if "macro_cat" in pois.columns:
        pois["macro_enc"] = pois["macro_cat"].astype("category").cat.codes + 1
    return pois


# ---------- Attach extra POI embeddings ----------
def _attach_npy_with_pk_map(pois: pd.DataFrame, npy_path: str, prefix: str):
    """
    Attach embeddings from an NPY via placekey mapping.
    Sidecar CSV required: <npy>.csv with a 'placekey' (or 'PLACEKEY') column.
    """
    arr = np.load(npy_path)  # [M, D]
    if arr.ndim != 2:
        raise ValueError(f"{npy_path} must be 2-D [N,D], got {arr.shape}")

    pk_csv = os.path.splitext(npy_path)[0] + ".csv"
    if not os.path.isfile(pk_csv):
        raise FileNotFoundError(f"Expected placekey list next to {npy_path}: {pk_csv}")

    pk_df = pd.read_csv(pk_csv)
    pk_col = "placekey" if "placekey" in pk_df.columns else ("PLACEKEY" if "PLACEKEY" in pk_df.columns else None)
    if pk_col is None:
        raise ValueError(f"{pk_csv} must contain a 'placekey' (or 'PLACEKEY') column")

    if len(pk_df) != arr.shape[0]:
        raise ValueError(f"Row count mismatch: {pk_csv} has {len(pk_df)} rows, but {npy_path} has {arr.shape[0]} rows")

    # placekey -> row index (first occurrence wins if duplicates)
    pk_series = pk_df[pk_col].astype(str)
    pk2row = {}
    for i, pk in enumerate(pk_series):
        if pk not in pk2row:
            pk2row[pk] = i

    D = arr.shape[1]
    cols = [f"{prefix}_{i}" for i in range(D)]
    mat = np.zeros((len(pois), D), dtype=np.float32)

    hits = 0
    for r, pk in enumerate(pois["placekey"].astype(str)):
        idx = pk2row.get(pk)
        if idx is not None:
            mat[r] = arr[idx].astype(np.float32)
            hits += 1

    for j, c in enumerate(cols):
        pois[c] = mat[:, j]

    hit_rate = 100.0 * hits / max(len(pois), 1)
    logging.info("Attached %s from %s (+sidecar CSV), hit-rate=%.2f%%", prefix, npy_path, hit_rate)
    return pois, cols


def _attach_table_by_placekey(pois: pd.DataFrame, path: str, key: str = "placekey", prefix: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    loader = pd.read_parquet if path.endswith(".parquet") else pd.read_csv
    extra = loader(path)
    assert key in extra.columns, f"{path} must contain '{key}'"
    numeric_cols = [c for c in extra.columns if c != key and pd.api.types.is_numeric_dtype(extra[c])]
    if prefix is not None:
        ren = {c: f"{prefix}_{i}" for i, c in enumerate(numeric_cols)}
        extra = extra[[key] + numeric_cols].rename(columns=ren)
        numeric_cols = [ren[c] for c in numeric_cols]
    out = pois.merge(extra[[key] + numeric_cols], on=key, how="left")
    out[numeric_cols] = out[numeric_cols].fillna(0).astype(np.float32)
    return out, numeric_cols


def _attach_from_plgot_anchor(
    pois: pd.DataFrame,
    plgot_poi_csv: str,
    npy_path: str,
    prefix: str = "dwell"
) -> Tuple[pd.DataFrame, List[str]]:
    anchor = pd.read_csv(plgot_poi_csv)
    if "placekey" not in anchor.columns and "PLACEKEY" in anchor.columns:
        anchor = anchor.rename(columns={"PLACEKEY": "placekey"})
    assert "placekey" in anchor.columns, f"{plgot_poi_csv} must include 'placekey'"

    arr = np.load(npy_path)
    assert arr.ndim == 2, f"{npy_path} must be 2-D [N,D], got {arr.shape}"
    assert len(arr) == len(anchor), f"{npy_path}: N={len(arr)} != len(anchor)={len(anchor)}"

    pk2row = dict(zip(anchor["placekey"].astype(str), range(len(anchor))))
    d = arr.shape[1]
    cols = [f"{prefix}_{i}" for i in range(d)]
    mat = np.zeros((len(pois), d), dtype=np.float32)
    for r, pk in enumerate(pois["placekey"].astype(str)):
        idx = pk2row.get(pk)
        if idx is not None:
            mat[r] = arr[idx].astype(np.float32)
    for i, c in enumerate(cols):
        pois[c] = mat[:, i]
    return pois, cols


def poi_feature_matrix_with_schema(
    pois_df: pd.DataFrame,
    feature_allowlist: Sequence[str],
    schema_path: str,
    reset_schema: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    feat_df = pois_df.reindex(columns=feature_allowlist, copy=True)
    for c in feature_allowlist:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df[feature_allowlist].copy()

    if os.path.exists(schema_path) and (not reset_schema):
        schema = load_json(schema_path)
        cols = schema.get("columns", feature_allowlist)
        for c in cols:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        feat_df = feat_df[cols]
    else:
        save_json(schema_path, {"columns": list(feat_df.columns)})

    feat_df = feat_df.fillna(0.0).astype(np.float32)
    return feat_df.values, feat_df.columns.tolist()


# ------------------------------
# CBG feature schema
# ------------------------------

def cbg_feature_matrix_with_schema(
    cbg_df: pd.DataFrame,
    schema_path: str,
    reset_schema: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    tmp = cbg_df.select_dtypes(include=[np.number]).copy()
    for drop in ["node_id"]:
        if drop in tmp.columns:
            tmp.drop(columns=[drop], inplace=True)
    if "CBG_ID" in tmp.columns:
        tmp.drop(columns=["CBG_ID"], inplace=True)

    if os.path.exists(schema_path) and (not reset_schema):
        schema = load_json(schema_path)
        cols = schema.get("columns", [c for c in tmp.columns])
        for c in cols:
            if c not in tmp.columns:
                tmp[c] = 0.0
        tmp = tmp[cols]
    else:
        save_json(schema_path, {"columns": list(tmp.columns)})

    tmp = tmp.fillna(0.0).astype(np.float32)
    return tmp.values, tmp.columns.tolist()


# ------------------------------
# Graph helpers
# ------------------------------

def parse_visitors_and_filter(
    pois: pd.DataFrame,
    fips_prefix: Optional[str] = None,
    min_visits: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    logging.info(
        "Parsing visitor_home_cbgs and filtering POIs… (fips_prefix=%s, min_visits=%s)",
        fips_prefix, min_visits
    )

    def _to_dict(v):
        if pd.isna(v):
            return {}
        if isinstance(v, dict):
            return {ensure_cbg12(k): int(vv) for k, vv in v.items() if ensure_cbg12(k)}
        s = str(v).strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {ensure_cbg12(k): int(vv) for k, vv in obj.items() if ensure_cbg12(k)}
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                out = {}
                for k, val in obj.items():
                    k12 = ensure_cbg12(k)
                    if k12 is not None:
                        try:
                            out[k12] = int(val)
                        except Exception:
                            out[k12] = 0
                return out
        except Exception:
            return {}
        return {}

    pois = pois.copy()
    pois["visitor_home_cbgs"] = pois["visitor_home_cbgs"].apply(_to_dict)

    if fips_prefix is None or str(fips_prefix).strip() == "":
        filt = pois["visitor_home_cbgs"].apply(lambda d: isinstance(d, dict) and len(d) > 0)
    else:
        prefix = str(fips_prefix)
        thr = int(min_visits)
        filt = pois["visitor_home_cbgs"].apply(
            lambda d: any(str(k).startswith(prefix) and int(v) > thr for k, v in d.items())
        )
    pois = pois[filt].copy()

    visitors_map = dict(zip(pois["placekey"], pois["visitor_home_cbgs"]))
    logging.info("Kept %d POIs after visitor filter.", len(pois))
    return pois, visitors_map


def load_label_visitors_map(
    csv_path: str,
    fips_prefix: Optional[str] = None,
    min_visits: int = 0,
) -> Dict[str, Dict[str, int]]:
    """TEMPORAL labels: load a (possibly different-week) POI CSV and return ONLY its
    visitors_map keyed by placekey. No feature encoding is done here — these records
    are used solely to build the visit (label) edges. Parsing/filtering is delegated to
    parse_visitors_and_filter so the label format is identical to same-week labels."""
    raw = pd.read_csv(csv_path)
    df = raw.copy()
    orig_cols = list(df.columns)
    df.columns = [c.lower() for c in df.columns]
    if "placekey" not in df.columns and "PLACEKEY" in orig_cols:
        df["placekey"] = raw["PLACEKEY"].astype(str)
    if "visitor_home_cbgs" not in df.columns and "VISITOR_HOME_CBGS" in orig_cols:
        df["visitor_home_cbgs"] = raw["VISITOR_HOME_CBGS"]
    if "visitor_home_cbgs" not in df.columns:
        raise ValueError(f"--label_poi_csv ({csv_path}) must contain 'visitor_home_cbgs' (or VISITOR_HOME_CBGS).")
    if "placekey" not in df.columns:
        raise ValueError(f"--label_poi_csv ({csv_path}) must contain 'placekey' (or PLACEKEY).")
    _, vmap = parse_visitors_and_filter(df, fips_prefix=fips_prefix, min_visits=min_visits)
    return vmap


def build_geometries(pois: pd.DataFrame, cbg: pd.DataFrame, shp_path: str, utm_epsg: int) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    logging.info("Building GeoDataFrames and reprojecting to EPSG:%d …", utm_epsg)

    # POIs
    if not {"latitude", "longitude"} <= set(pois.columns):
        raise ValueError("POI CSV must contain 'latitude' and 'longitude' columns.")
    poi_gdf = gpd.GeoDataFrame(
        pois.copy(),
        geometry=gpd.points_from_xy(pois["longitude"], pois["latitude"], crs="EPSG:4326"),
    ).to_crs(epsg=utm_epsg)
    poi_gdf["x"] = poi_gdf.geometry.x
    poi_gdf["y"] = poi_gdf.geometry.y

    # CBG polygons
    gdf_full = gpd.read_file(shp_path)
    if "GEOID" not in gdf_full.columns:
        for cand in ["geoid", "GEOID10", "GEOID20", "GEOIDFP"]:
            if cand in gdf_full.columns:
                gdf_full = gdf_full.rename(columns={cand: "GEOID"})
                break
    if "GEOID" not in gdf_full.columns:
        raise ValueError("CBG shapefile must have a 'GEOID' column.")

    gdf_full = gdf_full.to_crs(epsg=utm_epsg)
    cbg_ids = cbg["CBG_ID"].astype(str).str.zfill(12)
    gdf = gdf_full[gdf_full["GEOID"].astype(str).str.zfill(12).isin(cbg_ids)].copy()

    # Centroids for KNN
    gdf["centroid"] = gdf.geometry.centroid
    gdf["x"] = gdf["centroid"].x
    gdf["y"] = gdf["centroid"].y

    # ★ 关键：把 GeoDataFrame 行索引重置为 0..N-1，后续才能用按位置访问
    gdf = gdf.reset_index(drop=True)
    return poi_gdf, gdf


def sample_pois(
    poi_gdf: gpd.GeoDataFrame,
    pois_df: pd.DataFrame,
    n_pts: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    rng = np.random.default_rng(seed)
    max_pts = min(n_pts, len(poi_gdf))
    sample_pos = rng.choice(len(poi_gdf), size=max_pts, replace=False)
    poi_sample = poi_gdf.iloc[sample_pos].copy()
    # keep original row index for safe feature selection
    poi_sample["orig_row"] = poi_sample.index
    poi_sample = poi_sample.reset_index(drop=True)
    poi_sample["node_id"] = np.arange(len(poi_sample))

    if "poi_idx" not in poi_sample.columns:
        poi_sample["poi_idx"] = poi_sample.index
    poi_node_idx = dict(zip(poi_sample["poi_idx"], poi_sample["node_id"]))
    return poi_sample, poi_node_idx


def build_cbg_node_index(cbg: pd.DataFrame) -> Dict[str, int]:
    cbg = cbg.copy()
    cbg["node_id"] = np.arange(len(cbg))
    cbg_node_idx = dict(zip(cbg["CBG_ID"].astype(str).str.zfill(12), cbg["node_id"]))
    return cbg_node_idx


def fit_knn(gdf: gpd.GeoDataFrame, poi_sample: pd.DataFrame, k: int) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("Fitting KNN (k=%d) on CBG centroids…", k)
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
    nn.fit(gdf[["x", "y"]].values)
    dist_mat, idx_mat = nn.kneighbors(poi_sample[["x", "y"]].values, return_distance=True)
    return dist_mat.astype(np.float32), idx_mat.astype(np.int64)


def export_artifacts(output_dir: str,
                     poi_sample: pd.DataFrame,
                     idx_mat: np.ndarray,
                     gdf: gpd.GeoDataFrame,
                     cbg_node_idx: Dict[str, int],
                     run_name: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    poi_sample_path = os.path.join(output_dir, f"{run_name}_poi_sample.csv")
    mapping_path   = os.path.join(output_dir, f"{run_name}_poi_to_cbg_mapping.csv")
    cols_to_save = [c for c in poi_sample.columns if c != "geometry"]
    pd.DataFrame(poi_sample[cols_to_save]).to_csv(poi_sample_path, index=False)

    records = []
    for i, poi_row in pd.DataFrame(poi_sample).reset_index(drop=True).iterrows():
        placekey = str(poi_row.get("placekey", ""))
        poi_idx = int(poi_row.get("poi_idx", i))
        node_id = int(poi_row["node_id"])

        for rank, cbg_pos in enumerate(idx_mat[i], start=1):
            if int(cbg_pos) < 0:
                continue
            geoid = str(gdf.iloc[int(cbg_pos)]["GEOID"]).zfill(12)
            records.append({
                "node_id": node_id,
                "poi_idx": poi_idx,
                "placekey": placekey,
                "rank": rank,
                "cbg_index": int(cbg_pos),
                "CBG_ID": geoid,
                "cbg_node_id": cbg_node_idx.get(geoid, -1),
            })
    mapping_df = pd.DataFrame(records)
    mapping_df.to_csv(mapping_path, index=False)

    logging.info("✓ Saved %s (%d rows)", poi_sample_path, len(poi_sample))
    logging.info("✓ Saved %s (%d rows)", mapping_path, len(mapping_df))
    return mapping_df


def build_graph(poi_feat: np.ndarray, cbg_feat: np.ndarray) -> HeteroData:
    data = HeteroData()
    data["poi"].x = torch.tensor(poi_feat, dtype=torch.float32)
    data["cbg"].x = torch.tensor(cbg_feat, dtype=torch.float32)
    return data


def add_edges_belong(
    data: HeteroData,
    poi_sample: pd.DataFrame,
    cbg_node_idx: Dict[str, int],
) -> None:
    s_belong, d_belong = [], []
    for _, row in pd.DataFrame(poi_sample).iterrows():
        src = int(row["node_id"])
        cbg_code = ensure_cbg12(row.get("poi_cbg"))
        if cbg_code is None:
            continue
        dst = cbg_node_idx.get(cbg_code)
        if dst is None:
            continue
        s_belong.append(src)
        d_belong.append(dst)
    if s_belong:
        edge_index = torch.tensor([s_belong, d_belong], dtype=torch.long)
        edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float32)
        data[("poi", "belong", "cbg")].edge_index = edge_index
        data[("poi", "belong", "cbg")].edge_attr = edge_attr
        logging.info("Belong edges: %d", edge_index.shape[1])
    else:
        logging.warning("No 'belong' edges created (missing/filtered poi_cbg).")


def add_edges_knn(
    data: HeteroData,
    poi_sample: pd.DataFrame,
    idx_mat: np.ndarray,
    dist_mat: np.ndarray,
    gdf: gpd.GeoDataFrame,
    cbg_node_idx: Dict[str, int],
) -> None:
    gdf_row_flat = idx_mat.ravel()
    valid_mask = gdf_row_flat >= 0
    gdf_row_flat = gdf_row_flat[valid_mask]

    geoids_flat = gdf.iloc[gdf_row_flat]["GEOID"].astype(str).str.zfill(12).values
    knn_dst_flat = np.array([cbg_node_idx.get(code, -1) for code in geoids_flat], dtype=np.int64)

    knn_src_flat = np.repeat(np.arange(len(poi_sample), dtype=np.int64), idx_mat.shape[1])[valid_mask]
    knn_dist_km = (dist_mat.ravel()[valid_mask] / 1000.0).astype(np.float32)

    edge_index = torch.tensor([knn_src_flat, knn_dst_flat], dtype=torch.long)
    edge_attr = torch.tensor(knn_dist_km.reshape(-1, 1), dtype=torch.float32)
    data[("poi", "knn", "cbg")].edge_index = edge_index
    data[("poi", "knn", "cbg")].edge_attr = edge_attr
    logging.info("KNN edges: %d", edge_index.shape[1])


def add_edges_adjacent(data, gdf, cbg_node_idx, cbg_unique_pos):
    sub = gdf.iloc[cbg_unique_pos].copy().reset_index(drop=False)  # keep original row index
    sub.rename(columns={'index': '_gdf_row'}, inplace=True)

    # spatial index
    sidx = sub.sindex
    pairs = set()

    for r, geom in sub.geometry.items():
        # candidate neighbors via bbox
        for j in sidx.intersection(geom.bounds):
            if j == r:
                continue
            if not geom.touches(sub.geometry.iloc[j]):
                continue

            a_row = int(sub.iloc[r]['_gdf_row'])
            b_row = int(sub.iloc[j]['_gdf_row'])
            a_code = str(gdf.at[a_row, "GEOID"]).zfill(12)
            b_code = str(gdf.at[b_row, "GEOID"]).zfill(12)
            a_node = cbg_node_idx.get(a_code)
            b_node = cbg_node_idx.get(b_code)
            if a_node is not None and b_node is not None:
                pairs.add((a_node, b_node))

    if pairs:
        edge_index = torch.tensor(list(pairs), dtype=torch.long).T.contiguous()
        data[("cbg", "adjacent", "cbg")].edge_index = edge_index
        logging.info("Adjacency edges: %d", edge_index.shape[1])
    else:
        logging.warning("No adjacency edges created.")


def add_edges_visit(
    data: HeteroData,
    poi_sample: pd.DataFrame,
    idx_mat: np.ndarray,
    gdf: gpd.GeoDataFrame,
    visitors_map: Dict[str, Dict[str, int]],
    cbg_node_idx: Dict[str, int],
    drop_zero: bool = True,
) -> None:
    visit_src, visit_dst, visit_w = [], [], []

    num_poi = len(poi_sample)
    num_knn = idx_mat.shape[1] if idx_mat.ndim == 2 else 0

    for i in range(num_poi):
        row = pd.DataFrame(poi_sample).iloc[i]
        p_node = int(row["node_id"])
        placekey = str(row.get("placekey", ""))
        visitors = visitors_map.get(placekey, {})

        for j in range(num_knn):
            cbg_row = int(idx_mat[i, j])
            if cbg_row < 0:
                continue
            cbg_code = str(gdf.iloc[cbg_row]["GEOID"]).zfill(12)
            v_count = int(visitors.get(cbg_code, 0))
            c_node = cbg_node_idx.get(cbg_code)
            if c_node is None:
                continue
            visit_src.append(c_node)
            visit_dst.append(p_node)
            visit_w.append(v_count)

    if not visit_src:
        logging.warning("No visit edges created (missing visitors/CBGs).")
        return

    df = pd.DataFrame({"src": visit_src, "dst": visit_dst, "w": visit_w})
    sums = df.groupby("dst")["w"].transform("sum")
    df["w_norm"] = np.where(sums > 0, df["w"] / sums, 0.0)
    if drop_zero:
        df = df[df["w_norm"] > 0]

    edge_index = torch.tensor([df["src"].values, df["dst"].values], dtype=torch.long)
    edge_attr = torch.tensor(df["w_norm"].values.reshape(-1, 1), dtype=torch.float32)

    data[("cbg", "visit", "poi")].edge_index = edge_index
    data[("cbg", "visit", "poi")].edge_attr = edge_attr
    logging.info("Visit edges: %d (after zero-weight drop=%s)", edge_index.shape[1], drop_zero)


def drop_reverse_visit_edges(data: HeteroData) -> HeteroData:
    to_del = []
    for et in list(data.edge_types):
        s, rel, d = et
        if (s == 'poi') and (d == 'cbg') and ('visit' in rel):
            to_del.append(et)
    for et in to_del:
        del data[et]
    return data


def apply_transforms(
    data: HeteroData,
    to_undirected: bool,
    add_self_loops: bool,
    normalize_features: bool,
) -> HeteroData:
    if to_undirected:
        logging.info("Applying ToUndirected()…")
        data = T.ToUndirected()(data)
    if add_self_loops:
        logging.info("Applying AddSelfLoops()…")
        data = T.AddSelfLoops()(data)
    if normalize_features:
        logging.info("Applying NormalizeFeatures()…")
        data = T.NormalizeFeatures()(data)

    data = drop_reverse_visit_edges(data)
    return data


def validate_graph_indices(data: HeteroData) -> None:
    n_poi = data["poi"].x.shape[0]
    n_cbg = data["cbg"].x.shape[0]
    for et in data.edge_types:
        src_type, rel, dst_type = et
        edge_index = data[et].edge_index
        if src_type == "poi":
            assert int(edge_index[0].max()) < n_poi, f"{et} src index out of bounds"
        if dst_type == "poi":
            assert int(edge_index[1].max()) < n_poi, f"{et} dst index out of bounds"
        if src_type == "cbg":
            assert int(edge_index[0].max()) < n_cbg, f"{et} src index out of bounds"
        if dst_type == "cbg":
            assert int(edge_index[1].max()) < n_cbg, f"{et} dst index out of bounds"
    logging.info("✓ Edge index bounds validated.")


def save_graph(data: HeteroData, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    torch.save(data, out_path)
    logging.info("✓ Saved graph to %s", out_path)
    return out_path


# ------------------------------
# NEW: Integrate POI↔POI edges from PLGOT assets
# ------------------------------

def _load_plgot_anchor_placekeys(plgot_poi_csv: str) -> List[str]:
    df = pd.read_csv(plgot_poi_csv)
    if "placekey" not in df.columns and "PLACEKEY" in df.columns:
        df = df.rename(columns={"PLACEKEY": "placekey"})
    assert "placekey" in df.columns, f"{plgot_poi_csv} must include 'placekey'"
    return df["placekey"].astype(str).tolist()


def _add_poi_poi_edges_from_npz(
    data: HeteroData,
    relation_name: str,
    npz_path: Optional[str],
    poi_sample: pd.DataFrame,
    anchor_placekeys: List[str],
    attr_spec: Sequence[str],
) -> None:
    """将 PLGOT 的 npz（锚索引）映射到当前采样到的 poi_sample 节点，并写入 data。"""
    if not npz_path or (not os.path.isfile(npz_path)):
        logging.info("Skip %s: file missing (%s)", relation_name, npz_path)
        return

    z = np.load(npz_path)
    required = {"src", "dst"}
    if not required.issubset(set(z.files)):
        logging.warning("Skip %s: npz must contain %s. Found: %s", relation_name, required, list(z.files))
        return

    src_anchor = z["src"].astype(np.int64)
    dst_anchor = z["dst"].astype(np.int64)

    # placekey→node_id（仅限采样子集）
    pk_sample_to_node = dict(zip(pd.DataFrame(poi_sample)["placekey"].astype(str),
                                 pd.DataFrame(poi_sample)["node_id"].astype(int)))
    # 锚 placekey 索引 → 采样 node_id（-1 代表不在采样中）
    anchor2node = np.full((len(anchor_placekeys),), -1, dtype=np.int64)
    for i, pk in enumerate(anchor_placekeys):
        nid = pk_sample_to_node.get(pk, -1)
        anchor2node[i] = nid if nid is not None else -1

    src_nodes = anchor2node[src_anchor]
    dst_nodes = anchor2node[dst_anchor]
    mask = (src_nodes >= 0) & (dst_nodes >= 0)

    if mask.sum() == 0:
        logging.warning("No overlapping POI nodes for relation '%s' after sampling/filtering.", relation_name)
        return

    edge_index = np.stack([src_nodes[mask], dst_nodes[mask]], axis=0)

    if attr_spec:
        cols = []
        for key in attr_spec:
            if key not in z.files:
                raise ValueError(f"{npz_path} missing attr '{key}' required for relation '{relation_name}'")
            cols.append(z[key].astype(np.float32)[mask].reshape(-1, 1))
        edge_attr = np.concatenate(cols, axis=1)
    else:
        edge_attr = np.zeros((mask.sum(), 0), dtype=np.float32)

    data[("poi", relation_name, "poi")].edge_index = torch.tensor(edge_index, dtype=torch.long)
    if edge_attr.shape[1] > 0:
        data[("poi", relation_name, "poi")].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    logging.info("%s edges: %d  (attrs=%d)", relation_name, edge_index.shape[1], edge_attr.shape[1])


# ------------------------------
# Main
# ------------------------------

@dataclass
class PipelineConfig:
    poi_csv: str
    cbg_csv: str
    cbg_shp: str
    output_dir: str
    sample_size: int = 8000
    k_near: int = 20
    seed: int = 42
    utm_epsg: int = 32616
    make_adjacent: bool = True
    drop_zero_visit_edges: bool = True
    to_undirected: bool = True
    add_self_loops: bool = True
    normalize_features: bool = True
    save_enriched_pois: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="POI↔CBG graph pipeline (+ rich POI embeddings + optional POI↔POI edges)")

    # Positionals
    parser.add_argument("poi_csv", help="Path to POI CSV")
    parser.add_argument("cbg_csv", help="Path to CBG CSV")
    parser.add_argument("cbg_shp", help="Path to CBG shapefile (.shp)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--run_name", required=True, help="Name prefix for output files")

    # Schema/encoders
    parser.add_argument("--schema_dir", default=None, help="Dir to read/write feature schemas (default: <output_dir>)")
    parser.add_argument("--encoders_dir", default=None, help="Dir to read/write encoders (default: <output_dir>/encoders)")
    parser.add_argument("--reset_schema", action="store_true", help="Overwrite saved schemas (use for Week-1 only)")

    # Extra POI embeddings (can be combined)
    parser.add_argument("--poi_text_npy", default=None, help="Path to npy [N,D] aligned to filtered POIs")
    parser.add_argument("--poi_text_csv", default=None, help="CSV/Parquet with placekey + numeric text columns")
    parser.add_argument("--poi_dwell_npy", default=None, help="Path to npy [N,D_dwell] aligned to filtered POIs")
    parser.add_argument("--l2_norm_text", action="store_true", help="Row-wise L2 normalize text_* columns")

    # PLGOT artifacts (map via placekey) → for POI↔POI relations
    parser.add_argument("--plgot_poi_csv", default=None, help="Anchor CSV (e.g., .../athens_embedding/mydataset.csv)")
    parser.add_argument("--plgot_dwell_npy", default=None, help="dwell_feat.npy aligned to --plgot_poi_csv")
    parser.add_argument("--plgot_knn_npz",  default=None, help="knn_edges.npz (src,dst,dist,azim)")
    parser.add_argument("--plgot_time_npz", default=None, help="time_similarity_edges.npz (src,dst,weight)")
    parser.add_argument("--plgot_brand_npz", default=None, help="brand_week_edges.npz (src,dst,weight)")
    parser.add_argument("--plgot_cooc_npz", default=None, help="coocc_edges.npz (src,dst,weight)")

    # Behaviour
    parser.add_argument("--sample_size", type=int, default=8000, help="Max POIs to sample")
    parser.add_argument("--k", type=int, default=20, help="K nearest CBGs per POI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--utm_epsg", type=int, default=32616, help="UTM EPSG")
    parser.add_argument("--no_adjacent", action="store_true", help="Disable CBG adjacency edges")
    parser.add_argument("--keep_zero_visit_edges", action="store_true", help="Keep zero-weight visit edges")
    parser.add_argument("--no_to_undirected", action="store_true", help="Disable ToUndirected transform")
    parser.add_argument("--no_self_loops", action="store_true", help="Disable AddSelfLoops transform")
    parser.add_argument("--no_normalize_features", action="store_true", help="Disable NormalizeFeatures transform")
    parser.add_argument("--save_enriched_pois", action="store_true", help="Save enriched POIs as CSV")
    parser.add_argument("--verbosity", type=int, default=1, help="0=warn,1=info,2=debug")

    # Output graph filename override
    parser.add_argument("--out_graph_name", default=None,
                        help="Custom graph filename (.pt). If not set, defaults to knn_<run_name>_hetero_data.pt")

    # Optional filter for POIs by FIPS coverage in visitors
    parser.add_argument("--fips_prefix", default=None,
                        help="FIPS prefix (e.g., '13121'). Keep POIs with visitors from CBGs whose GEOID starts with this prefix.")
    parser.add_argument("--min_visits", type=int, default=0,
                        help="Minimum visits per matching CBG when --fips_prefix is set (strictly greater than this value).")

    # ---- TEMPORAL (leakage-free) labels ----
    # Features and ALL edges (POI features, POI-POI, POI-CBG candidates, CBG) come from --poi_csv (week t-1).
    # ONLY the visit (label) edges come from --label_poi_csv (week t), matched to the sampled POIs by placekey.
    # If --label_poi_csv is not set, behaviour is IDENTICAL to build_graph_final.py.
    parser.add_argument("--label_poi_csv", default=None,
                        help="Build visit (label) edges from THIS POI CSV (e.g. week t) instead of --poi_csv (week t-1). "
                             "Matched to sampled POIs by placekey. Unset => same-week labels (original behaviour).")
    parser.add_argument("--label_min_visits", type=int, default=None,
                        help="min_visits used when parsing --label_poi_csv. Defaults to --min_visits.")

    args = parser.parse_args()
    setup_logging(args.verbosity)

    output_dir = args.output_dir
    schema_dir = args.schema_dir or output_dir
    encoders_dir = args.encoders_dir or os.path.join(output_dir, "encoders")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(schema_dir, exist_ok=True)
    os.makedirs(encoders_dir, exist_ok=True)

    cfg = PipelineConfig(
        poi_csv=args.poi_csv,
        cbg_csv=args.cbg_csv,
        cbg_shp=args.cbg_shp,
        output_dir=output_dir,
        sample_size=args.sample_size,
        k_near=args.k,
        seed=args.seed,
        utm_epsg=args.utm_epsg,
        make_adjacent=not args.no_adjacent,
        drop_zero_visit_edges=not args.keep_zero_visit_edges,
        to_undirected=not args.no_to_undirected,
        add_self_loops=not args.no_self_loops,
        normalize_features=not args.no_normalize_features,
        save_enriched_pois=args.save_enriched_pois,
    )

    # ------------------ Load POI/CBG ------------------
    logging.info("Loading POI: %s", cfg.poi_csv)
    raw_pois = pd.read_csv(cfg.poi_csv)
    pois = raw_pois.copy()
    orig_cols = list(pois.columns)
    pois.columns = [c.lower() for c in pois.columns]
    # Backward-compatible fallbacks
    if "placekey" not in pois.columns and "PLACEKEY" in orig_cols:
        pois["placekey"] = raw_pois["PLACEKEY"].astype(str)
    if "visitor_home_cbgs" not in pois.columns and "VISITOR_HOME_CBGS" in orig_cols:
        pois["visitor_home_cbgs"] = raw_pois["VISITOR_HOME_CBGS"]
    if "latitude" not in pois.columns and "LATITUDE" in orig_cols:
        pois["latitude"] = raw_pois["LATITUDE"].astype(float)
    if "longitude" not in pois.columns and "LONGITUDE" in orig_cols:
        pois["longitude"] = raw_pois["LONGITUDE"].astype(float)

    cbg = pd.read_csv(cfg.cbg_csv)
    if "CBG_ID" not in cbg.columns and "cbg_id" in cbg.columns:
        cbg = cbg.rename(columns={"cbg_id": "CBG_ID"})

    # ------------------ Enrich POIs ------------------
    pois = encode_categories(pois)
    pois = compute_open_hours_features(pois)

    if "visitor_home_cbgs" not in pois.columns:
        raise ValueError("POI CSV must contain 'visitor_home_cbgs' (or VISITOR_HOME_CBGS)")
    pois, visitors_map = parse_visitors_and_filter(
        pois,
        fips_prefix=args.fips_prefix,
        min_visits=args.min_visits,
    )

    # ---- TEMPORAL labels: choose which week supplies the visit (label) edges ----
    if args.label_poi_csv:
        lbl_min = args.label_min_visits if args.label_min_visits is not None else args.min_visits
        logging.info("TEMPORAL MODE: visit labels from %s (label_min_visits=%s); "
                     "features/edges remain from %s", args.label_poi_csv, lbl_min, cfg.poi_csv)
        label_visitors_map = load_label_visitors_map(
            args.label_poi_csv, fips_prefix=args.fips_prefix, min_visits=lbl_min,
        )
        n_overlap = sum(1 for pk in pois["placekey"].astype(str) if str(pk) in label_visitors_map)
        logging.info("Label-week visitors_map: %d POIs | overlap with feature-week kept POIs: %d",
                     len(label_visitors_map), n_overlap)
    else:
        label_visitors_map = visitors_map  # same-week labels (original behaviour)

    # ---------- Extra POI embeddings (AFTER filtering) ----------
    extra_cols: List[str] = []

    if args.poi_text_csv:
        logging.info("Merging text/geo embeddings from table: %s", args.poi_text_csv)
        pois, cols = _attach_table_by_placekey(pois, args.poi_text_csv, key="placekey", prefix="text")
        extra_cols += cols
    if args.poi_text_npy:
        logging.info("Attaching text embeddings from npy (+sidecar .csv): %s", args.poi_text_npy)
        pois, cols = _attach_npy_with_pk_map(pois, args.poi_text_npy, prefix="text")
        extra_cols += cols

    # Dwell via PLGOT anchor or npy+csv sidecar
    if args.plgot_poi_csv and args.plgot_dwell_npy:
        logging.info("Attaching dwell via PLGOT anchor: %s | %s", args.plgot_poi_csv, args.plgot_dwell_npy)
        pois, cols = _attach_from_plgot_anchor(pois, args.plgot_poi_csv, args.plgot_dwell_npy, prefix="dwell")
        extra_cols += cols
    elif args.poi_dwell_npy:
        logging.info("Attaching dwell embeddings from npy (+sidecar .csv): %s", args.poi_dwell_npy)
        pois, cols = _attach_npy_with_pk_map(pois, args.poi_dwell_npy, prefix="dwell")
        extra_cols += cols

    if args.l2_norm_text:
        tcols = [c for c in extra_cols if c.startswith("text_")]
        if tcols:
            X = pois[tcols].values.astype(np.float32)
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            pois[tcols] = (X / n).astype(np.float32)

    # ------------------ Geometries ------------------
    poi_gdf, gdf = build_geometries(pois, cbg, cfg.cbg_shp, cfg.utm_epsg)

    # ------------------ Sampling + KNN ------------------
    # （本版本始终随机采样；已移除复用旧采样/旧映射的功能）
    poi_sample, _ = sample_pois(poi_gdf, pois, cfg.sample_size, cfg.seed)
    dist_mat, idx_mat = fit_knn(gdf, poi_sample, cfg.k_near)

    # ------------------ Feature matrices ------------------
    def _rows_for_features(src_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
        if "orig_row" in sample_df.columns:
            return src_df.iloc[pd.Series(sample_df["orig_row"]).astype(int).tolist()].copy()
        return src_df.iloc[sample_df.index]

    poi_schema_path = os.path.join(schema_dir, "poi_feature_schema.json")
    poi_feat, poi_feature_names = poi_feature_matrix_with_schema(
        pois_df=_rows_for_features(pois, poi_sample),
        feature_allowlist=POI_FEATURE_ALLOWLIST + extra_cols,
        schema_path=poi_schema_path,
        reset_schema=args.reset_schema,
    )
    # 保存特征列清单
    features_txt = os.path.join(output_dir, f"{args.run_name}_poi_features.txt")
    with open(features_txt, "w", encoding="utf-8") as f:
        f.write("POI feature columns used for node features (order-locked):\n")
        for name in poi_feature_names:
            f.write(f"{name}\n")

    cbg_node_idx = build_cbg_node_index(cbg)
    cbg_schema_path = os.path.join(schema_dir, "cbg_feature_schema.json")
    cbg_feat, cbg_feature_names = cbg_feature_matrix_with_schema(
        cbg_df=cbg, schema_path=cbg_schema_path, reset_schema=args.reset_schema,
    )

    # ------------------ Export artifacts ------------------
    _ = export_artifacts(output_dir, poi_sample, idx_mat, gdf, cbg_node_idx, args.run_name)

    # ------------------ Assemble graph ------------------
    data = build_graph(poi_feat, cbg_feat)
    add_edges_belong(data, poi_sample, cbg_node_idx)
    add_edges_knn(data, poi_sample, idx_mat, dist_mat, gdf, cbg_node_idx)

    cbg_unique_pos = np.unique(idx_mat[idx_mat >= 0])
    if cfg.make_adjacent:
        add_edges_adjacent(data, gdf, cbg_node_idx, cbg_unique_pos)

    add_edges_visit(
        data,
        poi_sample,
        idx_mat,
        gdf,
        label_visitors_map,   # TEMPORAL: labels from --label_poi_csv (== visitors_map when unset)
        cbg_node_idx,
        drop_zero=cfg.drop_zero_visit_edges,
    )
    if args.label_poi_csv:
        _vp = data[("cbg", "visit", "poi")]
        _n_lbl = int(_vp.edge_index[1].unique().numel()) if ("edge_index" in _vp) else 0
        logging.info("TEMPORAL: %d / %d sampled POIs have a label-week visit distribution "
                     "(others get an all-zero target and are masked out during training/eval).",
                     _n_lbl, len(poi_sample))

    # Optional: add POI↔POI edges from PLGOT artifacts
    if args.plgot_poi_csv:
        anchor_placekeys = _load_plgot_anchor_placekeys(args.plgot_poi_csv)
        _add_poi_poi_edges_from_npz(
            data, "geo_knn", args.plgot_knn_npz, poi_sample, anchor_placekeys, ["dist", "azim"]
        )
        _add_poi_poi_edges_from_npz(
            data, "time_sim", args.plgot_time_npz, poi_sample, anchor_placekeys, ["weight"]
        )
        _add_poi_poi_edges_from_npz(
            data, "brand", args.plgot_brand_npz, poi_sample, anchor_placekeys, ["weight"]
        )
        _add_poi_poi_edges_from_npz(
            data, "cooc", args.plgot_cooc_npz, poi_sample, anchor_placekeys, ["weight"]
        )

    # ------------------ Transforms / Validate / Save ------------------
    data = apply_transforms(data, cfg.to_undirected, cfg.add_self_loops, cfg.normalize_features)
    validate_graph_indices(data)
    graph_name = args.out_graph_name or f"knn_{args.run_name}_hetero_data.pt"
    graph_path = save_graph(data, output_dir, filename=graph_name)

    # Save enriched POIs (optional; only sampled subset)
    if cfg.save_enriched_pois:
        enriched_path = os.path.join(output_dir, f"{args.run_name}_pois_enriched.csv")
        if "orig_row" in poi_sample.columns:
            pois.iloc[pd.Series(poi_sample["orig_row"]).astype(int).tolist()].to_csv(enriched_path, index=False)
        else:
            pois.iloc[poi_sample.index].to_csv(enriched_path, index=False)
        logging.info("✓ Saved enriched POIs to %s", enriched_path)

    save_json(os.path.join(output_dir, f"{args.run_name}_poi_feature_names.json"), {"columns": poi_feature_names})
    save_json(os.path.join(output_dir, f"{args.run_name}_cbg_feature_names.json"), {"columns": cbg_feature_names})

    # ------------------ PRINT summaries for quick sanity-check ------------------
    # （这些 print 是给你在控制台快速确认用的；logging 也会有更详细记录）
    print("\n================== SUMMARY (PRINT) ==================")
    print(f"POIs (sampled): {int(data['poi'].x.shape[0])}")
    print(f"CBGs (from CBG table): {int(data['cbg'].x.shape[0])}")
    print(f"POI feature dim: {int(data['poi'].x.size(1))} | CBG feature dim: {int(data['cbg'].x.size(1))}")
    print("\n-- Edge types & sizes (after transforms) --")
    for et in sorted(list(data.edge_types)):
        ei = data[et].edge_index
        ea = getattr(data[et], 'edge_attr', None)
        shape = None if ea is None else tuple(ea.shape)
        print(f"{et!s:35s} edges={ei.shape[1]:8d}  edge_attr={shape}")
    # 简要提示是否包含 POI↔POI 关系
    poi_poi_rels = [et for et in data.edge_types if et[0] == 'poi' and et[2] == 'poi']
    if poi_poi_rels:
        print("\n[Info] POI↔POI relations present:")
        for et in sorted(poi_poi_rels):
            ea = getattr(data[et], 'edge_attr', None)
            print(f"  {et}  edge_attr_dim={None if ea is None else ea.size(1)}")
    else:
        print("\n[Info] No POI↔POI relations found in this graph.")

    print(f"\nGraph saved to: {graph_path}")
    print("=====================================================\n")

    # 同时记录到日志（方便保存到文件）
    logging.info("=== Summary ===")
    logging.info("POIs (sampled): %d", data["poi"].x.shape[0])
    logging.info("CBGs (from CBG table): %d", data["cbg"].x.shape[0])
    for et, ei in data.edge_index_dict.items():
        logging.info("%s edges: %d (shape=%s)", et, ei.shape[1], tuple(ei.shape))
    logging.info("Graph saved: %s", graph_path)
    logging.info("POI feature dim: %d | CBG feature dim: %d", data["poi"].x.size(1), data["cbg"].x.size(1))


if __name__ == "__main__":
    main()


# python /home/lp43319/projects/GNN/visitgnn/build_graph_final.py \
#   /home/lp43319/projects/GNN/visitgnn/data/Fulton_POI_week1_with_macro.csv \
#   /home/lp43319/projects/GNN/visitgnn/data/Fulton_cbg.csv \
#   /home/lp43319/projects/GNN/visitgnn/data/tl_2019_13_bg/tl_2019_13_bg.shp \
#   /home/lp43319/projects/GNN/visitgnn/output/fulton_w1 \
#   --run_name fulton_w1 \
#   --k 544 --sample_size 8000 \
#   --poi_text_csv /home/lp43319/projects/GNN/visitgnn/poi_embedding/mydataset_geo.parquet \
#   --plgot_poi_csv /home/lp43319/projects/GNN/visitgnn/poi_embedding/mydataset.csv \
#   --plgot_dwell_npy /home/lp43319/projects/GNN/visitgnn/poi_embedding/dwell_feat.npy \
#   --plgot_knn_npz  /home/lp43319/projects/GNN/visitgnn/poi_embedding/knn_edges.npz \
#   --plgot_time_npz /home/lp43319/projects/GNN/visitgnn/poi_embedding/time_similarity_edges.npz \
#   --plgot_brand_npz /home/lp43319/projects/GNN/visitgnn/poi_embedding/brand_week_edges.npz \
#   --l2_norm_text \
#   --fips_prefix 13121 --min_visits 10 \
#   --out_graph_name fulton_w1_hetero_with_text_and_poi_edges.pt \
#   --verbosity 2

