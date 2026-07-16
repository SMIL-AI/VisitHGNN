#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baselines_gravity_radiation.py  -  classic spatial-interaction baselines.

Addresses R3#9 (which explicitly names gravity and radiation models) / R2#3 / R1.
Both models predict, for each POI, a probability distribution over the SAME K
candidate origin-CBGs used by VisitHGNN, and write a preds CSV in the
inferencecopy format (poi_node_id, rank_in_knn, cbg_node_id, pred_prob) so the
SAME evaluate_and_plot.py scores them identically.

Models
------
GRAVITY:    p(cbg | POI) ∝ Pop(cbg) / d(cbg,POI)^beta
            beta is fit on the validation POIs by minimising mean per-POI KL
            (pass --gravity_beta <float> to fix it instead).

RADIATION:  Simini et al. (2012) radiation model with intervening opportunities.
            For origin cbg i and destination POI j:
                T_ij ∝ m_i^2 / [ (m_i + s_ij) (m_i + n_j + s_ij) ]
            where  m_i  = Pop(cbg i)
                   s_ij = total population of CBGs strictly closer to i than
                          the POI is to i (intervening opportunities; self
                          excluded)
                   n_j  = destination "mass" of the POI (a per-POI constant; it
                          cancels as a prefactor but enters the denominator).
                          Default = median CBG population (low sensitivity since
                          s_ij typically dominates); override with --radiation_nj.
            Distances for the radius use the graph's km candidate distances
            (Euclidean UTM, same projection that defined the candidate set);
            CBG-CBG distances for s_ij are computed from CBG centroids projected
            to the same UTM zone, so both are consistent.

Inputs
------
--graph_path   : the *_split.pt graph (candidate edges + km distances + visit
                 weights + train/val/test_idx). Same file you evaluate on.
--mapping_csv  : <run>_poi_to_cbg_mapping.csv  (must contain cbg_node_id, CBG_ID)
--cbg_csv      : Fulton_cbg.csv  (must contain CBG_ID, Pop_Tot, longitude, latitude)
--out_dir      : where to write the preds CSVs.

Run (cluster venv):
    python baselines_gravity_radiation.py \
        --graph_path  .../fulton_w1f_w2l/fulton_w1f_w2l_split.pt \
        --mapping_csv .../fulton_w1f_w2l/fulton_w1f_w2l_poi_to_cbg_mapping.csv \
        --cbg_csv     /home/lp43319/projects/GNN/visitgnn/data/Fulton_cbg.csv \
        --out_dir     /home/lp43319/projects/GNN/visitgnn/output/baselines/spatial
Then score each preds CSV with the SAME flags as the model, e.g.
    python evaluate_and_plot.py --graph_path <graph_split.pt> \
        --preds_csv <out>/<stem>_gravity_preds.csv --out_dir <out> \
        --split test --ndcg_k 50 --recall_k 5 --match_by auto
"""

import argparse
import os
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Pure-numpy cores (no torch) -- unit-testable                               #
# --------------------------------------------------------------------------- #
def normalize_rows(score, valid):
    """Row-normalise a [N,K] score matrix over valid entries -> probabilities.
    Rows whose valid scores sum to 0 stay all-zero."""
    s = np.where(valid, score, 0.0).astype(np.float64)
    denom = s.sum(axis=1, keepdims=True)
    out = np.divide(s, denom, out=np.zeros_like(s), where=denom > 0)
    return out


def gravity_scores(pop_mat, dist_mat, valid, beta, dist_floor):
    """Gravity score Pop / d^beta on a [N,K] candidate grid (un-normalised)."""
    d = np.maximum(dist_mat, dist_floor)
    sc = np.where(valid, pop_mat / np.power(d, beta), 0.0)
    return sc


def twosfca_scores(pop_mat, dist_mat, valid, knn, beta, dist_floor):
    """2SFCA-flow score (Luo & Wang two-step floating catchment, demand-allocation form).

    Reduces, for the per-POI origin distribution, to gravity reweighted by the
    inverse accessibility of each origin CBG:
        p(cbg_i | POI_j) ∝ D_i * w(d_ij) / A_i,     w(d) = d^-beta
        R_j = S_j / Σ_{i∈cand(j)} D_i w(d_ij)          (step 1; supply S_j = 1)
        A_i = Σ_{j': i∈cand(j')} R_j' w(d_ij')         (step 2; CBG accessibility)
    The constant supply S_j cancels in the per-POI normalisation, but the
    two-step COMPETITION enters through A_i: an origin CBG near many POIs has
    high A_i and is down-weighted (its demand is split across competitors) —
    the distinctive 2SFCA effect that plain gravity lacks. Returns un-normalised
    scores on the [N,K] candidate grid."""
    d = np.maximum(dist_mat, dist_floor)
    w = np.where(valid, np.power(d, -beta), 0.0)                       # [N,K] distance decay
    Dpot = (pop_mat * w).sum(axis=1)                                  # [N] weighted demand potential
    R = np.divide(1.0, Dpot, out=np.zeros_like(Dpot), where=Dpot > 0)  # [N] step-1 ratio (S_j=1)

    cells = np.argwhere(valid & (knn >= 0))
    sc = np.zeros_like(w)
    if cells.size == 0:
        return sc
    ci, cj = cells[:, 0], cells[:, 1]
    nmax = int(knn[ci, cj].max()) + 1
    A = np.zeros(nmax, dtype=np.float64)                              # accessibility per CBG node id
    np.add.at(A, knn[ci, cj], R[ci] * w[ci, cj])                     # step 2 scatter-add
    Ai = A[knn[ci, cj]]
    num = pop_mat[ci, cj] * w[ci, cj]                                # D_i * w(d_ij)
    sc[ci, cj] = np.divide(num, Ai, out=np.zeros_like(num), where=Ai > 0)
    return sc


def mean_poi_kl(pred, true_p, poi_mask):
    """Mean per-POI KL( true || pred ) over POIs in poi_mask that have mass.
    Matches evaluate_and_plot: per-POI normalise both over valid candidates,
    kl = sum_c true*log(true/pred); average over scored POIs."""
    eps = 1e-12
    rows = np.where(poi_mask)[0]
    kls = []
    for i in rows:
        t = true_p[i]
        if t.sum() <= 0:
            continue
        p = pred[i]
        tn = t / t.sum()
        pn = p / p.sum() if p.sum() > 0 else p
        m = tn > 0
        # candidates with true>0 but pred==0 -> infinite KL; guard with eps
        kls.append(float(np.sum(tn[m] * np.log((tn[m] + eps) / (pn[m] + eps)))))
    return float(np.mean(kls)) if kls else float('inf')


def _score_grid(model, pop_mat, dist_mat, valid, knn, beta, dist_floor):
    if model == 'gravity':
        return gravity_scores(pop_mat, dist_mat, valid, beta, dist_floor)
    if model == '2sfca':
        return twosfca_scores(pop_mat, dist_mat, valid, knn, beta, dist_floor)
    raise ValueError(f"unknown decay model: {model}")


def fit_beta(model, pop_mat, dist_mat, valid, knn, true_p, val_mask, dist_floor,
             grid=None, refine=True):
    """Pick beta minimising mean val-POI KL for a decay model ('gravity' or '2sfca');
    coarse grid then a local refine around the best point."""
    if grid is None:
        grid = np.round(np.arange(0.2, 4.01, 0.2), 3)
    best_b, best_kl, trace = None, float('inf'), []

    def ev(b):
        nonlocal best_b, best_kl
        pred = normalize_rows(_score_grid(model, pop_mat, dist_mat, valid, knn, b, dist_floor), valid)
        kl = mean_poi_kl(pred, true_p, val_mask)
        trace.append((float(b), kl))
        if kl < best_kl:
            best_kl, best_b = kl, float(b)

    for b in grid:
        ev(b)
    if refine and best_b is not None:
        for b in np.round(np.linspace(best_b - 0.2, best_b + 0.2, 9), 4):
            if b > 0:
                ev(b)
    return best_b, best_kl, sorted(trace)


def project_lonlat_to_xy(lon, lat, epsg):
    """Project lon/lat -> planar metres. Uses pyproj if available, else a local
    equirectangular approximation about the data centroid (accurate to <0.5% at
    county scale, which is all radiation's intervening-opportunity ranking needs)."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
        x, y = tr.transform(lon, lat)
        return np.asarray(x, float), np.asarray(y, float), "pyproj"
    except Exception:
        R = 6371000.0
        lat0 = np.deg2rad(np.nanmean(lat))
        x = np.deg2rad(lon) * R * np.cos(lat0)
        y = np.deg2rad(lat) * R
        return x, y, "equirectangular-fallback"


def build_cbg_sorted(xy, pop):
    """For each CBG, sorted distances (km) to all CBGs and cumulative population.
    Returns sorted_d[Ncbg,Ncbg], cumpop[Ncbg,Ncbg] (aligned)."""
    n = xy.shape[0]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000.0  # km
    order = np.argsort(d, axis=1)
    sorted_d = np.take_along_axis(d, order, axis=1)
    sorted_pop = pop[order]
    cumpop = np.cumsum(sorted_pop, axis=1)
    return sorted_d, cumpop


def radiation_scores(knn_univ, dist_mat, valid, pop_univ, sorted_d, cumpop, n_j):
    """Radiation score on the [N,K] candidate grid (un-normalised).
    knn_univ[N,K] = universe index of each candidate origin CBG (-1 invalid)."""
    N, K = knn_univ.shape
    sc = np.zeros((N, K), dtype=np.float64)
    # group candidate cells by origin-CBG to batch the searchsorted per origin
    valid_cells = np.argwhere(valid & (knn_univ >= 0))
    if valid_cells.size == 0:
        return sc
    origins = knn_univ[valid_cells[:, 0], valid_cells[:, 1]]
    radii = dist_mat[valid_cells[:, 0], valid_cells[:, 1]]
    for u in np.unique(origins):
        sel = origins == u
        r = radii[sel]
        # number of CBGs strictly closer than radius r (sorted_d[u] is ascending)
        pos = np.searchsorted(sorted_d[u], r, side='left')
        pos = np.clip(pos, 0, cumpop.shape[1])
        cum = np.where(pos > 0, cumpop[u][np.clip(pos - 1, 0, cumpop.shape[1] - 1)], 0.0)
        m_i = float(pop_univ[u])
        s_ij = cum - m_i                      # exclude self (always closest, d=0)
        s_ij = np.maximum(s_ij, 0.0)
        denom = (m_i + s_ij) * (m_i + n_j + s_ij)
        val = np.where(denom > 0, (m_i * m_i) / denom, 0.0)
        rows = valid_cells[sel, 0]
        cols = valid_cells[sel, 1]
        sc[rows, cols] = val
    return sc


# --------------------------------------------------------------------------- #
#  Graph loading (torch only needed here; imported lazily for testability)    #
# --------------------------------------------------------------------------- #
def extract_candidates(graph_path, K):
    """Return knn_idx[N,K] (cbg node ids, -1 pad, nearest-first),
    dist_km[N,K], true_p[N,K], and split index arrays. Mirrors
    train_pp_optimized.build_targets so candidates/order match exactly."""
    import torch
    data = torch.load(graph_path, weights_only=False)

    e_knn = data[('poi', 'knn', 'cbg')]
    src = e_knn.edge_index[0].cpu().numpy()
    dst = e_knn.edge_index[1].cpu().numpy()
    dkm = (e_knn.edge_attr.view(-1).cpu().numpy().astype(float)
           if getattr(e_knn, 'edge_attr', None) is not None
           else np.zeros_like(src, dtype=float))
    df = pd.DataFrame({'poi': src, 'cbg': dst, 'dist': dkm}).sort_values(['poi', 'dist'])

    e_vis = data[('cbg', 'visit', 'poi')]
    vmap = {(int(p), int(c)): float(w) for c, p, w in zip(
        e_vis.edge_index[0].cpu().numpy(), e_vis.edge_index[1].cpu().numpy(),
        e_vis.edge_attr.view(-1).cpu().numpy())}

    N = int(data['poi'].num_nodes)
    knn = np.full((N, K), -1, dtype=np.int64)
    dist = np.zeros((N, K), dtype=np.float64)
    true_p = np.zeros((N, K), dtype=np.float64)
    for poi, g in df.groupby('poi'):
        tk = g.head(K)
        cbgs = tk['cbg'].to_numpy()
        n = len(cbgs)
        knn[poi, :n] = cbgs
        dist[poi, :n] = tk['dist'].to_numpy()
        ws = np.array([vmap.get((int(poi), int(c)), 0.0) for c in cbgs], dtype=float)
        s = ws.sum()
        if s > 0:
            true_p[poi, :n] = ws / (s + 1e-8)

    def fetch(key):
        t = getattr(data, key, None)
        if t is None:
            try:
                t = data[key]
            except Exception:
                t = None
        if t is None and isinstance(data, dict):
            t = data.get(key, None)
        try:
            import torch as _t
            if isinstance(t, _t.Tensor):
                return t.long().view(-1).cpu().numpy()
        except Exception:
            pass
        return None

    tr, va, te = fetch('train_idx'), fetch('val_idx'), fetch('test_idx')
    if tr is None or te is None:
        order = np.random.default_rng(42).permutation(N)
        n_tr, n_va = int(0.7 * N), int(0.15 * N)
        tr, va, te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
        print("[spatial] WARNING: stored split not found; recomputed with seed=42.")
    return knn, dist, true_p, tr, (va if va is not None else np.array([], int)), te


# --------------------------------------------------------------------------- #
def _norm_id(s):
    """Normalise a CBG identifier to a bare string (strip, drop trailing .0)."""
    s = str(s).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def main():
    ap = argparse.ArgumentParser(description="Gravity & radiation spatial-interaction baselines.")
    ap.add_argument('--graph_path', required=True)
    ap.add_argument('--mapping_csv', required=True, help="poi_to_cbg_mapping.csv (needs cbg_node_id, CBG_ID)")
    ap.add_argument('--cbg_csv', required=True, help="Fulton_cbg.csv (CBG_ID, Pop_Tot, longitude, latitude)")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--models', default='gravity,2sfca,radiation', help="comma list: gravity,2sfca,radiation")
    ap.add_argument('--gravity_beta', default='fit', help="'fit' (min val-KL) or a float")
    ap.add_argument('--twosfca_beta', default='fit', help="'fit' (min val-KL) or a float (2SFCA decay)")
    ap.add_argument('--dist_floor_km', type=float, default=0.05, help="floor on candidate distance for gravity")
    ap.add_argument('--radiation_nj', default='median_pop', help="'median_pop' or a float (POI destination mass)")
    ap.add_argument('--utm_epsg', type=int, default=32616, help="UTM zone EPSG for CBG centroid projection")
    ap.add_argument('--cbgid_col', default='CBG_ID')
    ap.add_argument('--pop_col', default='Pop_Tot')
    ap.add_argument('--lon_col', default='longitude')
    ap.add_argument('--lat_col', default='latitude')
    ap.add_argument('--node_cbgid_col', default='CBG_ID', help="CBG-id column name in the mapping CSV")
    ap.add_argument('--node_id_col', default='cbg_node_id', help="cbg-node-id column name in the mapping CSV")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    # ---- candidates / targets / splits from the graph ----
    print(f"[spatial] loading graph: {args.graph_path}")
    knn, dist, true_p, tr_idx, va_idx, te_idx = extract_candidates(args.graph_path, args.k)
    N, K = knn.shape
    valid = knn >= 0
    val_mask = np.zeros(N, bool); val_mask[va_idx] = True
    print(f"[spatial] N={N} POIs, K={K}; split train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    # ---- cbg_node_id -> CBG_ID (from mapping) ----
    mp = pd.read_csv(args.mapping_csv, usecols=[args.node_id_col, args.node_cbgid_col]).drop_duplicates()
    node2cbgid = {int(r[args.node_id_col]): _norm_id(r[args.node_cbgid_col]) for _, r in mp.iterrows()}

    # ---- CBG universe (ALL cbgs: needed for radiation intervening pop) ----
    cbg = pd.read_csv(args.cbg_csv, usecols=[args.cbgid_col, args.pop_col, args.lon_col, args.lat_col]).copy()
    cbg[args.cbgid_col] = cbg[args.cbgid_col].map(_norm_id)
    cbg = cbg.drop_duplicates(subset=[args.cbgid_col]).reset_index(drop=True)
    cbgid2uni = {cid: i for i, cid in enumerate(cbg[args.cbgid_col].tolist())}
    pop_univ = cbg[args.pop_col].to_numpy(dtype=float)
    pop_univ = np.nan_to_num(pop_univ, nan=0.0)
    print(f"[spatial] CBG universe: {len(cbg)} CBGs; median pop = {np.median(pop_univ):.1f}")

    # candidate CBG -> universe index, and candidate population matrix
    uni_of_node = {nid: cbgid2uni.get(cid, -1) for nid, cid in node2cbgid.items()}
    knn_univ = np.full((N, K), -1, dtype=np.int64)
    pop_mat = np.zeros((N, K), dtype=np.float64)
    miss = 0
    nz = np.argwhere(valid)
    for i, j in nz:
        nid = int(knn[i, j])
        u = uni_of_node.get(nid, -1)
        knn_univ[i, j] = u
        if u >= 0:
            pop_mat[i, j] = pop_univ[u]
        else:
            miss += 1
    cover = 1.0 - miss / max(1, int(valid.sum()))
    print(f"[spatial] candidate->population coverage = {cover:.4f} "
          f"({miss} of {int(valid.sum())} candidate cells unmatched)")
    if cover < 0.9:
        print("[spatial] WARNING: low coverage. Check that CBG_ID formats match between "
              "the mapping CSV and the CBG CSV (leading zeros / dtype).")

    stem = os.path.splitext(os.path.basename(args.graph_path))[0]
    written = []

    # ============================ GRAVITY ============================ #
    if 'gravity' in models:
        if str(args.gravity_beta).lower() == 'fit':
            beta, val_kl, trace = fit_beta(
                'gravity', pop_mat, dist, valid, knn, true_p, val_mask, args.dist_floor_km)
            print(f"[gravity] fitted beta = {beta:.3f}  (val KL = {val_kl:.4f})")
            near = [f"{b:.2f}:{kl:.4f}" for b, kl in trace if abs(b - beta) <= 0.4]
            print(f"[gravity]   val-KL near optimum: {'  '.join(near)}")
        else:
            beta = float(args.gravity_beta)
            print(f"[gravity] using fixed beta = {beta:.3f}")
        pred = normalize_rows(gravity_scores(pop_mat, dist, valid, beta, args.dist_floor_km), valid)
        out_csv = _write_preds(pred, knn, valid, args.out_dir, f"{stem}_gravity")
        print(f"[gravity] wrote {out_csv}  (test KL on this file via evaluate_and_plot)")
        written.append(out_csv)

    # ============================ 2SFCA ============================ #
    if '2sfca' in models:
        if str(args.twosfca_beta).lower() == 'fit':
            beta2, vkl2, tr2 = fit_beta(
                '2sfca', pop_mat, dist, valid, knn, true_p, val_mask, args.dist_floor_km)
            print(f"[2sfca] fitted beta = {beta2:.3f}  (val KL = {vkl2:.4f})")
            near = [f"{b:.2f}:{kl:.4f}" for b, kl in tr2 if abs(b - beta2) <= 0.4]
            print(f"[2sfca]   val-KL near optimum: {'  '.join(near)}")
        else:
            beta2 = float(args.twosfca_beta)
            print(f"[2sfca] using fixed beta = {beta2:.3f}")
        pred = normalize_rows(twosfca_scores(pop_mat, dist, valid, knn, beta2, args.dist_floor_km), valid)
        out_csv = _write_preds(pred, knn, valid, args.out_dir, f"{stem}_2sfca")
        print(f"[2sfca] wrote {out_csv}  (2SFCA-flow = gravity reweighted by inverse CBG accessibility)")
        written.append(out_csv)

    # ============================ RADIATION ============================ #
    if 'radiation' in models:
        x, y, proj = project_lonlat_to_xy(cbg[args.lon_col].to_numpy(),
                                          cbg[args.lat_col].to_numpy(), args.utm_epsg)
        xy = np.column_stack([x, y])
        print(f"[radiation] projected CBG centroids via {proj} (EPSG:{args.utm_epsg})")
        sorted_d, cumpop = build_cbg_sorted(xy, pop_univ)
        if str(args.radiation_nj).lower() == 'median_pop':
            n_j = float(np.median(pop_univ[pop_univ > 0])) if (pop_univ > 0).any() else 1.0
        else:
            n_j = float(args.radiation_nj)
        print(f"[radiation] destination mass n_j = {n_j:.1f}")
        sc = radiation_scores(knn_univ, dist, valid, pop_univ, sorted_d, cumpop, n_j)
        pred = normalize_rows(sc, valid)
        out_csv = _write_preds(pred, knn, valid, args.out_dir, f"{stem}_radiation")
        print(f"[radiation] wrote {out_csv}")
        written.append(out_csv)

    print("\n[spatial] DONE. Score each with the SAME flags as the model, e.g.:")
    for f in written:
        print(f"  python evaluate_and_plot.py --graph_path {args.graph_path} \\\n"
              f"      --preds_csv {f} --out_dir {args.out_dir} \\\n"
              f"      --split test --ndcg_k 50 --recall_k 5 --match_by auto")


def _write_preds(pred, knn, valid, out_dir, stem):
    """Flatten a [N,K] prediction grid to the inferencecopy preds CSV."""
    N, K = knn.shape
    flat = valid.ravel()
    poi_id = np.repeat(np.arange(N), K)[flat]
    rank = (np.tile(np.arange(K), N)[flat] + 1)
    cbg_id = knn.ravel()[flat]
    prob = pred.ravel()[flat]
    out = (pd.DataFrame({'poi_node_id': poi_id, 'rank_in_knn': rank,
                         'cbg_node_id': cbg_id, 'pred_prob': prob})
           .sort_values(['poi_node_id', 'rank_in_knn']).reset_index(drop=True))
    path = os.path.join(out_dir, f"{stem}_preds.csv")
    out.to_csv(path, index=False)
    return path


if __name__ == '__main__':
    main()
