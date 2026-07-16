#!/usr/bin/env bash
# =============================================================================
# run_multiseed.sh   (R3#10, approach i: FIXED graph, vary only the TRAINING seed)
#
# For ONE graph + the tuned config, runs  train -> inference(geo+time) -> eval
# for several seeds, each into <out_base>/seed_<s>/. Run it TWICE:
#   - once on the leaky (w1->w1) graph
#   - once on the leakage-free (w1->w2) graph
# Then compare with aggregate_seeds.py.
#
# Usage:
#   bash run_multiseed.sh <graph_split.pt> <out_base> <mapping_csv> <poi_sample_csv> [seed ...]
#
# Example (leakage-free, 5 seeds):
#   bash run_multiseed.sh \
#     /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_split.pt \
#     /home/lp43319/projects/GNN/visitgnn/output/MS_w1w2 \
#     /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_poi_to_cbg_mapping.csv \
#     /home/lp43319/projects/GNN/visitgnn/output/fulton_w1f_w2l/fulton_w1f_w2l_poi_sample.csv \
#     0 1 2 3 4
#
# Env overrides: REPO=<dir with canonical .py>  EPOCHS=1000  DEVICE=cuda:0
# =============================================================================
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: bash run_multiseed.sh <graph_split.pt> <out_base> <mapping_csv> <poi_sample_csv> [seed ...]" >&2
  exit 1
fi
GRAPH="$1"; OUT_BASE="$2"; MAP="$3"; SAMPLE="$4"; shift 4
SEEDS=("$@"); [ "${#SEEDS[@]}" -eq 0 ] && SEEDS=(0 1 2 3 4)

REPO="${REPO:-/home/lp43319/projects/GNN/2026GNN}"   # where the canonical .py live
EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda:0}"
STEM="$(basename "$GRAPH" .pt)"

echo "graph    = $GRAPH"
echo "out_base = $OUT_BASE"
echo "seeds    = ${SEEDS[*]}   | epochs=$EPOCHS | device=$DEVICE | repo=$REPO"
cd "$REPO"

for s in "${SEEDS[@]}"; do
  OD="$OUT_BASE/seed_$s"
  echo "================= SEED $s -> $OD ================="
  mkdir -p "$OD/prediction"

  # 1) train (tuned config; dims left at defaults d_hidden=64 d_poi=128 d_cbg=256 k=50)
  python train_pp_optimized.py \
    --graph_path "$GRAPH" --out_dir "$OD" \
    --epochs "$EPOCHS" --seed "$s" \
    --gate_init 0.12 --gate_anneal cosine --gate_anneal_warmup 80 \
    --learnable_rel_scale --rel_gates \
    --edge_temp_geo 1.2 --edge_temp_time 1.2 \
    --pp_l2 5e-6 --dropout 0.15 --weight_decay 3e-5
  test -f "$OD/best_visit_gnn.pt" || { echo "FATAL: no checkpoint for seed $s" >&2; exit 2; }

  # 2) params.json (fixed dims; inferencecopy also auto-overrides d_hidden from the ckpt)
  cat > "$OD/params.json" <<JSON
{"k": 50, "d_hidden": 64, "d_cbg": 256, "d_poi": 128, "dropout": 0.15}
JSON

  # 3) inference (geo+time, NO random brand)
  python inferencecopy.py \
    --graph_path "$GRAPH" --ckpt_path "$OD/best_visit_gnn.pt" \
    --params_json "$OD/params.json" --mapping_csv "$MAP" \
    --output_dir "$OD/prediction" --device "$DEVICE" \
    --force_with_poi_poi --poi_poi_modes geo_knn,time_sim
  test -f "$OD/prediction/${STEM}_preds.csv" || { echo "FATAL: no preds for seed $s" >&2; exit 3; }

  # 4) evaluate (writes ${STEM}_preds_test_with_gt.csv); default viz=worst (light, fast)
  python evaluate_and_plot.py \
    --graph_path "$GRAPH" \
    --preds_csv "$OD/prediction/${STEM}_preds.csv" \
    --poi_map_csv "$MAP" --poi_sample_csv "$SAMPLE" \
    --out_dir "$OD/prediction/eva" \
    --k 50 --split test --ndcg_k 50 --recall_k 5 --match_by auto
  echo "---- seed $s done: $OD/prediction/${STEM}_preds_test_with_gt.csv"
done

echo "ALL SEEDS DONE under: $OUT_BASE"
echo "with_gt files: $OUT_BASE/seed_*/prediction/${STEM}_preds_test_with_gt.csv"
