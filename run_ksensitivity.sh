#!/usr/bin/env bash
# =============================================================================
# run_ksensitivity.sh   (R3#8 candidate coverage + R2#4 K-sensitivity, MULTI-SEED)
#
# For ONE graph + the tuned config, sweeps the candidate-set size K and, for each
# K, runs several seeds:  train(--k K) -> inference(k=K) -> eval  into
# <out_base>/K_<K>/seed_<s>/. The graph is unchanged (it already stores all 544
# candidates; K only slices the top-K nearest in build_targets) and the model is
# K-agnostic (it scores POI-CBG pairs), so only --k and params.json's "k" vary.
#
# Pair each K's metrics with the coverage at that K (candidate_coverage.py) and
# aggregate across seeds with ksensitivity_table.py.
#
# Usage:
#   bash run_ksensitivity.sh <graph_split.pt> <out_base> <mapping_csv> <poi_sample_csv> [K ...]
#   (seeds via env:  SEEDS="0 1 2")
#
# Example (full sweep, 3 seeds each, on the leakage-free graph):
#   SEEDS="0 1 2" bash run_ksensitivity.sh \
#     /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_split.pt \
#     /home/lp43319/projects/GNN/visitgnn/output/KS_w1w2 \
#     /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_poi_to_cbg_mapping.csv \
#     /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_poi_sample.csv \
#     25 50 75 100 150 200 300 400 544
#
# Env overrides: REPO=<dir with canonical .py>  EPOCHS=1000  DEVICE=cuda:0  SEEDS="0 1 2"
# =============================================================================
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: bash run_ksensitivity.sh <graph_split.pt> <out_base> <mapping_csv> <poi_sample_csv> [K ...]" >&2
  echo "       seeds via env, e.g.  SEEDS=\"0 1 2\" bash run_ksensitivity.sh ... 25 50 100 200 544" >&2
  exit 1
fi
GRAPH="$1"; OUT_BASE="$2"; MAP="$3"; SAMPLE="$4"; shift 4
KS=("$@"); [ "${#KS[@]}" -eq 0 ] && KS=(25 50 75 100 150 200 300 400 544)

REPO="${REPO:-/home/lp43319/projects/GNN/2026GNN}"
EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda:0}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
STEM="$(basename "$GRAPH" .pt)"

echo "graph    = $GRAPH"
echo "out_base = $OUT_BASE"
echo "Ks       = ${KS[*]}"
echo "seeds    = ${SEEDS[*]}   | epochs=$EPOCHS | device=$DEVICE | repo=$REPO"
echo "total runs = $(( ${#KS[@]} * ${#SEEDS[@]} ))"
cd "$REPO"

for K in "${KS[@]}"; do
  for s in "${SEEDS[@]}"; do
    OD="$OUT_BASE/K_$K/seed_$s"
    echo "================= K=$K seed=$s -> $OD ================="
    mkdir -p "$OD/prediction"

    # 1) train at candidate-set size K (tuned config)
    python train_pp_optimized.py \
      --graph_path "$GRAPH" --out_dir "$OD" \
      --epochs "$EPOCHS" --seed "$s" --k "$K" \
      --gate_init 0.12 --gate_anneal cosine --gate_anneal_warmup 80 \
      --learnable_rel_scale --rel_gates \
      --edge_temp_geo 1.2 --edge_temp_time 1.2 \
      --pp_l2 5e-6 --dropout 0.15 --weight_decay 3e-5
    test -f "$OD/best_visit_gnn.pt" || { echo "FATAL: no checkpoint for K=$K seed=$s" >&2; exit 2; }

    # 2) params.json with matching k
    cat > "$OD/params.json" <<JSON
{"k": $K, "d_hidden": 64, "d_cbg": 256, "d_poi": 128, "dropout": 0.15}
JSON

    # 3) inference at the same K (geo+time, NO random brand conv)
    python inferencecopy.py \
      --graph_path "$GRAPH" --ckpt_path "$OD/best_visit_gnn.pt" \
      --params_json "$OD/params.json" --mapping_csv "$MAP" \
      --output_dir "$OD/prediction" --device "$DEVICE" \
      --force_with_poi_poi --poi_poi_modes geo_knn,time_sim
    test -f "$OD/prediction/${STEM}_preds.csv" || { echo "FATAL: no preds for K=$K seed=$s" >&2; exit 3; }

    # 4) evaluate (native-@K KL/MAE; NDCG@50/Recall@5 fixed cutoffs)
    python evaluate_and_plot.py \
      --graph_path "$GRAPH" \
      --preds_csv "$OD/prediction/${STEM}_preds.csv" \
      --poi_map_csv "$MAP" --poi_sample_csv "$SAMPLE" \
      --out_dir "$OD/prediction/eva" \
      --k "$K" --split test --ndcg_k 50 --recall_k 5 --match_by auto
    echo "---- K=$K seed=$s done"
  done
done

echo
echo "ALL DONE under: $OUT_BASE"
echo "with_gt files: $OUT_BASE/K_*/seed_*/prediction/${STEM}_preds_test_with_gt.csv"
echo "Next: python ksensitivity_table.py --root $OUT_BASE --graph_path $GRAPH --stem $STEM"
