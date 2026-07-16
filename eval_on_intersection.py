#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_on_intersection.py
------------------------
Compare two models (leaky w1->w1  vs  leakage-free w1->w2) on the COMMON set of
test POIs that have a valid (non-zero) target in BOTH evaluations. This removes
the "different test subset" confound (the leaky model is scored on all test POIs,
the leakage-free model only on POIs that also have a week-2 label).

All metrics replicate evaluate_and_plot.py EXACTLY (in-candidate, per-POI
normalised): KL, MAE, Top-1, NDCG@k, Recall@k.

It also prints a week1-vs-week2 TARGET-distribution shift on the same POIs
(mean entropy, mean top-1 mass). This probes the second confound: if the week-2
targets are intrinsically more concentrated, a lower KL is achievable on them
regardless of leakage.

Inputs are the `*_preds_test_with_gt.csv` files written by evaluate_and_plot.py.
POIs are matched by `poi_node_id`: the two graphs share the SAME seeded 8000-POI
sample, so node_id <-> placekey is identical across them.

Usage:
  python eval_on_intersection.py \
    --leaky_csv <Train_w1_clean>/prediction/..._preds_test_with_gt.csv \
    --free_csv  <Train_w1f_w2l_tuned>/prediction/..._preds_test_with_gt.csv \
    --ndcg_k 50 --recall_k 5
"""
import argparse
import numpy as np
import pandas as pd

EPS = 1e-9


def load_with_gt(path):
    """Return {poi_node_id: (pred[np.ndarray], gt[np.ndarray])} over each POI's
    candidates. Only rows with a (non-NaN) ground_truth are kept, which drops any
    non-test POIs (their gt is NaN after the left-merge in evaluate_and_plot)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    poi_c = pick(['poi_node_id', 'poi', 'poi_id', 'poi_idx', 'node_id'])
    # tolerate pandas merge suffixes. In evaluate_and_plot's left-merge dfp.merge(df_gt),
    # the authoritative graph-derived target (what its own metrics use) is the RIGHT
    # table's column -> 'ground_truth_y'. Prefer it over the preds-side '_x'.
    prob_c = pick(['pred_prob', 'pred_prob_x', 'pred_prob_y', 'prob', 'prediction', 'pred', 'score'])
    gt_c = pick(['ground_truth', 'ground_truth_y', 'ground_truth_x', 'gt', 'true', 'target'])
    cbg_c = pick(['cbg_node_id', 'cbg', 'cbg_id', 'cbg_idx'])
    if poi_c is None or prob_c is None or gt_c is None:
        raise KeyError(f"{path}: need poi / pred / ground_truth columns; got {list(df.columns)}")

    ren = {poi_c: 'poi', prob_c: 'pred', gt_c: 'gt'}
    if cbg_c is not None:
        ren[cbg_c] = 'cbg'
    df = df.rename(columns=ren)

    df = df[~df['gt'].isna()].copy()          # keep test POIs only
    df['poi'] = df['poi'].astype(int)
    df['pred'] = df['pred'].astype(float)
    df['gt'] = df['gt'].astype(float)

    out = {}
    for poi, g in df.groupby('poi'):
        out[int(poi)] = (g['pred'].to_numpy(dtype=float), g['gt'].to_numpy(dtype=float))
    return out


def _norm(v):
    v = np.clip(v, 0.0, None)
    s = v.sum()
    return v / s if s > EPS else v


def metrics_on(pois, store, ndcg_k, recall_k):
    """Replicates evaluate_and_plot.metrics_basic / ndcg_at_k / recall_at_k,
    restricted to `pois`. Each POI's pred/gt are normalised over its candidates."""
    kls, abs_errs, ndcgs, recalls = [], [], [], []
    top1_hits, n = 0, 0
    for poi in pois:
        p_raw, t_raw = store[poi]
        if t_raw.sum() <= EPS:
            continue
        p = _norm(p_raw)
        t = _norm(t_raw)
        n += 1
        # KL = sum_cand t * log(t/p)   (mean over POIs)
        kls.append(float(np.sum(t * (np.log(t + EPS) - np.log(p + EPS)))))
        # MAE = |p - t| over all valid candidates (flat mean)
        abs_errs.extend(np.abs(p - t).tolist())
        # Top-1
        if int(np.argmax(p)) == int(np.argmax(t)):
            top1_hits += 1
        # NDCG@k
        k = max(1, min(int(ndcg_k), len(p)))
        order = np.argsort(-p)[:k]
        rel = t[order]
        dcg = float(np.sum(rel / np.log2(np.arange(2, 2 + len(rel)))))
        ideal = np.sort(t)[::-1][:k]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, 2 + len(ideal)))))
        if idcg > EPS:
            ndcgs.append(dcg / idcg)
        # Recall@k = target mass captured by top-k predicted candidates
        rk = max(1, min(int(recall_k), len(p)))
        topk = np.argsort(-p)[:rk]
        recalls.append(float(np.sum(t[topk])))
    return {
        'n_poi': n,
        'KL': float(np.mean(kls)) if kls else float('nan'),
        'MAE': float(np.mean(abs_errs)) if abs_errs else float('nan'),
        'Top1': (top1_hits / n) if n else float('nan'),
        f'NDCG@{ndcg_k}': float(np.mean(ndcgs)) if ndcgs else float('nan'),
        f'Recall@{recall_k}': float(np.mean(recalls)) if recalls else float('nan'),
    }


def target_stats(pois, store):
    """Mean entropy (nats) and mean top-1 mass of the (normalised) target dists."""
    ents, tops = [], []
    for poi in pois:
        t = _norm(store[poi][1])
        if t.sum() <= EPS:
            continue
        ents.append(float(-np.sum(t * np.log(t + EPS))))
        tops.append(float(np.max(t)))
    return (float(np.mean(ents)) if ents else float('nan'),
            float(np.mean(tops)) if tops else float('nan'))


def main():
    ap = argparse.ArgumentParser(
        description="Compare leaky vs leakage-free on the common valid-POI intersection.")
    ap.add_argument("--leaky_csv", required=True,
                    help="*_preds_test_with_gt.csv for the leaky (w1->w1) model")
    ap.add_argument("--free_csv", required=True,
                    help="*_preds_test_with_gt.csv for the leakage-free (w1->w2) model")
    ap.add_argument("--ndcg_k", type=int, default=50)
    ap.add_argument("--recall_k", type=int, default=5)
    args = ap.parse_args()

    leaky = load_with_gt(args.leaky_csv)
    free = load_with_gt(args.free_csv)

    leaky_valid = {p for p, (pr, gt) in leaky.items() if gt.sum() > EPS}
    free_valid = {p for p, (pr, gt) in free.items() if gt.sum() > EPS}
    inter = sorted(leaky_valid & free_valid)

    print(f"leaky      test POIs (valid): {len(leaky_valid)}")
    print(f"leakage-free test POIs (valid): {len(free_valid)}")
    print(f"intersection (valid in BOTH):   {len(inter)}")
    if not inter:
        print("No common valid POIs — check that the two graphs share the same POI sampling.")
        return
    print()

    mL = metrics_on(inter, leaky, args.ndcg_k, args.recall_k)
    mF = metrics_on(inter, free, args.ndcg_k, args.recall_k)

    keys = ['KL', 'MAE', 'Top1', f'NDCG@{args.ndcg_k}', f'Recall@{args.recall_k}']
    print(f"=== Metrics on the SAME {len(inter)} POIs (intersection) ===")
    print(f"{'metric':<14}{'leaky w1->w1':>16}{'free w1->w2':>16}{'delta(free-leaky)':>20}")
    for k in keys:
        a, b = mL[k], mF[k]
        print(f"{k:<14}{a:>16.4f}{b:>16.4f}{(b - a):>20.4f}")
    print()

    e1, m1 = target_stats(inter, leaky)   # week-1 targets
    e2, m2 = target_stats(inter, free)    # week-2 targets
    print("=== Target-distribution shift on the same POIs (week-shift confound probe) ===")
    print(f"{'':<22}{'week1 target':>16}{'week2 target':>16}")
    print(f"{'mean entropy (nats)':<22}{e1:>16.4f}{e2:>16.4f}")
    print(f"{'mean top-1 mass':<22}{m1:>16.4f}{m2:>16.4f}")
    print()
    print("Read: if the week-2 targets have LOWER entropy / HIGHER top-1 mass, they are")
    print("intrinsically more concentrated -> a lower KL is achievable on them regardless of")
    print("leakage, i.e. part of any KL gap is a week-shift effect, not a leakage effect.")


if __name__ == "__main__":
    main()
