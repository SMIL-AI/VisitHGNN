#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
candidate_coverage.py  -  R3#8: does the top-K nearest-CBG candidate set capture
the real visit origins?

For every POI the model only predicts over its K nearest candidate CBGs. This
script quantifies how much of the POI's actual (in-region) visit mass falls
within those K nearest CBGs, and how that coverage grows with K — the data-side
justification for the K=50 choice.

Method (self-contained from the graph .pt)
------------------------------------------
The ('cbg','visit','poi') edge weights are row-normalised per POI over the
candidate CBGs at build time (w_norm = visits / Σ visits-to-candidates), so they
already express each candidate's FRACTION of the POI's in-region visit mass.
Ranking each POI's candidates by distance (the ('poi','knn','cbg') edge_attr, in
km) and taking the cumulative sum of w_norm gives:

    coverage_K(POI) = Σ_{the K nearest candidates} w_norm
                    = fraction of that POI's in-region visit mass captured by its
                      K nearest CBGs.

IMPORTANT — this is only meaningful if the graph stores the FULL CBG ranking per
POI (built with `--k <num_cbg>`, e.g. 544). The script auto-detects the number of
candidates per POI and warns if it is small (then coverage saturates at the
stored K by construction and a rebuild with the full ranking is needed).

This measures IN-REGION top-K coverage. Visits originating OUTSIDE the study
region's CBGs are not represented in the graph at all; that out-of-region share
is a separate framing point (it can be quantified from the raw
`visitor_home_cbgs` column — see the note printed at the end).

Run (cluster venv; login node fine, no CUDA):
    python candidate_coverage.py \
        --graph_path /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_split.pt \
        --out_csv    /home/lp43319/projects/GNN/visitgnn/output/baselines/coverage_curve.csv
"""

import argparse
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  torch-free core (unit-testable)                                            #
# --------------------------------------------------------------------------- #
def coverage_from_edges(knn_src, knn_dst, knn_dist, vis_cbg, vis_poi, vis_w,
                        n_poi, n_cbg, ks):
    """Compute coverage-vs-K and effective-K statistics from raw edge arrays.

    Returns a dict with the per-K coverage stats, effective-K to reach
    {90,95,99}% coverage, and the distribution of visited-candidate counts."""
    knn = pd.DataFrame({'poi': knn_src, 'cbg': knn_dst, 'dist': knn_dist}).sort_values(['poi', 'dist'])
    cand_counts = knn.groupby('poi').size()
    cmax = int(cand_counts.max())
    cmed = int(cand_counts.median())

    vis = {}
    for c, p, w in zip(vis_cbg, vis_poi, vis_w):
        vis.setdefault(int(p), {})[int(c)] = float(w)

    ks = sorted({int(k) for k in ks if int(k) <= cmax})
    covK = {k: [] for k in ks}
    eff = {90: [], 95: [], 99: []}
    nvis = []

    for poi, g in knn.groupby('poi'):
        vmap = vis.get(int(poi))
        if not vmap:
            continue                              # unlabeled POI (no visits)
        ordered = g['cbg'].to_numpy()             # candidates sorted by distance
        w = np.array([vmap.get(int(c), 0.0) for c in ordered], dtype=float)
        tot = w.sum()
        if tot <= 0:
            continue
        cum = np.cumsum(w) / tot                  # cumulative in-region coverage by rank
        nvis.append(int((w > 0).sum()))
        L = len(cum)
        for k in ks:
            covK[k].append(float(cum[min(k, L) - 1]))
        for thr in (90, 95, 99):
            idx = int(np.searchsorted(cum, thr / 100.0))
            eff[thr].append(idx + 1 if idx < L else L)

    return {
        'n_poi_total': int(n_poi),
        'n_cbg': int(n_cbg),
        'cand_per_poi_median': cmed,
        'cand_per_poi_max': cmax,
        'n_labeled': len(nvis),
        'ks': ks,
        'covK': {k: np.asarray(v) for k, v in covK.items()},
        'eff': {t: np.asarray(v) for t, v in eff.items()},
        'nvis': np.asarray(nvis),
    }


def _extract(graph_path):
    import torch
    data = torch.load(graph_path, weights_only=False)
    ek = data[('poi', 'knn', 'cbg')]
    src = ek.edge_index[0].cpu().numpy()
    dst = ek.edge_index[1].cpu().numpy()
    dist = (ek.edge_attr.view(-1).cpu().numpy()
            if getattr(ek, 'edge_attr', None) is not None else np.zeros_like(src, float))
    ev = data[('cbg', 'visit', 'poi')]
    vc = ev.edge_index[0].cpu().numpy()
    vp = ev.edge_index[1].cpu().numpy()
    vw = ev.edge_attr.view(-1).cpu().numpy()
    return src, dst, dist, vc, vp, vw, int(data['poi'].num_nodes), int(data['cbg'].num_nodes)


def main():
    ap = argparse.ArgumentParser(description="Candidate-set visit-mass coverage vs K (R3#8).")
    ap.add_argument('--graph_path', required=True, help="graph .pt (ideally built with --k = num_cbg)")
    ap.add_argument('--ks', default='5,10,20,30,40,50,75,100,150,200,300,544')
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    print(f"[cov] loading graph: {args.graph_path}")
    src, dst, dist, vc, vp, vw, N, ncbg = _extract(args.graph_path)
    ks_in = [int(x) for x in args.ks.split(',') if x.strip()]
    R = coverage_from_edges(src, dst, dist, vc, vp, vw, N, ncbg, ks_in)

    print(f"[cov] N={R['n_poi_total']} POIs, {R['n_cbg']} CBGs; "
          f"candidates/POI: median={R['cand_per_poi_median']} max={R['cand_per_poi_max']}")
    if R['cand_per_poi_max'] < R['n_cbg'] * 0.8:
        print(f"[cov] *** NOTE: the graph stores ~{R['cand_per_poi_median']} candidates/POI, far below "
              f"all {R['n_cbg']} CBGs. Visit weights are normalised over these candidates, so coverage "
              f"saturates at K={R['cand_per_poi_max']} by construction and CANNOT reveal in-region "
              f"origins missed beyond it. Rebuild the graph with --k {R['n_cbg']} and re-run for a valid "
              f"coverage curve. (Numbers below are still correct but bounded by K={R['cand_per_poi_max']}.)")

    print(f"[cov] labeled POIs used: {R['n_labeled']}")
    nv = R['nvis']
    print(f"[cov] distinct visited candidates per POI: mean={nv.mean():.1f} median={np.median(nv):.0f} "
          f"p90={np.percentile(nv, 90):.0f} max={int(nv.max())}")

    print("\n[cov] mean IN-REGION visit-mass coverage by K:")
    print("       K     mean    median   %POIs<80%")
    rows = []
    for k in R['ks']:
        a = R['covK'][k]
        rows.append((k, float(a.mean()), float(np.median(a)), float(np.mean(a < 0.8))))
        print(f"     {k:>4d}   {a.mean():.4f}   {np.median(a):.4f}    {np.mean(a < 0.8) * 100:5.1f}")

    print("\n[cov] nearest-CBG count needed to reach a coverage level (per POI):")
    for thr in (90, 95, 99):
        e = R['eff'][thr]
        print(f"     {thr}%:  mean K={e.mean():.1f}  median={np.median(e):.0f}  p90={np.percentile(e, 90):.0f}")

    if 50 in R['covK']:
        a = R['covK'][50]
        print(f"\n[cov] >>> at K=50: mean coverage = {a.mean():.4f}, median = {np.median(a):.4f}, "
              f"{np.mean(a >= 0.95) * 100:.1f}% of POIs ≥95% covered, "
              f"{np.mean(a >= 0.99) * 100:.1f}% ≥99%")

    if args.out_csv:
        pd.DataFrame(rows, columns=['K', 'mean_coverage', 'median_coverage', 'frac_POIs_below_0.8']) \
            .to_csv(args.out_csv, index=False)
        print(f"\n[cov] wrote {args.out_csv}")

    print("\n[cov] (This is IN-REGION top-K coverage. To also report the share of total visits that "
          "originate OUTSIDE the study CBGs, parse the raw `visitor_home_cbgs` column — ask if you "
          "want that added.)")


if __name__ == '__main__':
    main()
