# VisitHGNN

**Heterogeneous Graph Neural Networks for Modeling Point-of-Interest Visit Patterns.**

VisitHGNN predicts, for each point of interest (POI), a probability distribution over
candidate origin Census Block Groups (CBGs) — i.e. *where the visitors to a POI come from* —
using a heterogeneous graph that combines POI and CBG nodes, multiple POI–POI relations,
POI–CBG membership/proximity edges, and CBG–CBG adjacency, trained with a masked
Kullback–Leibler objective over a distance-ranked candidate set.

This repository contains the code accompanying the paper *"VisitHGNN: Heterogeneous Graph
Neural Networks for Modeling Point-of-Interest Visit Patterns"* (under review at
*Applied Intelligence*).

---

## Contents

**Core pipeline**

| Script | Purpose |
|---|---|
| `build_graph_final.py` | Build the heterogeneous graph (nodes, features, edges) |
| `build_graph_temporal.py` | Leakage-free graph builder: week-1 features/edges, week-2 labels (`--label_poi_csv`) |
| `data_split.py` | Deterministic train/val/test split (seed 42) |
| `hyper.py` | Optuna hyperparameter search |
| `train_pp_optimized.py` | Train the VisitHGNN model |
| `inferencecopy.py` | Run inference → per-POI candidate predictions |
| `evaluate_and_plot.py` | Compute metrics (KL, MAE, Top-1, NDCG@k, Recall@k) and figures |

**Baselines**

| Script | Purpose |
|---|---|
| `mlp_baseline.py` | Pairwise-MLP baseline |
| `baseline_gbdt.py` | Gradient-boosted-tree baseline (feature-matched to VisitHGNN) |
| `baseline_hetero_gnn.py` | RGCN / HAN baselines (and distance-augmented variants) |
| `baselines_gravity_radiation.py` | Gravity / 2SFCA / radiation spatial-interaction models |
| `baseline_table.py` | Aggregate baselines into a common-test-POI comparison table |

**Evaluation & analysis**

| Script | Purpose |
|---|---|
| `eval_on_intersection.py` | Shared metric utilities (scored on a common POI set; matches the main table) |
| `aggregate_seeds.py` | Aggregate multi-seed runs (mean ± std) |
| `crosscity_table.py` | Cross-city comparison (independent retrain + zero-shot transfer) |
| `candidate_coverage.py` | Candidate-set coverage analysis |
| `ksensitivity_table.py` | Candidate-set-size (K) sensitivity table |
| `plot_ksensitivity.py` | K-sensitivity figure |
| `train_ablation.py` | Ablation matrix (relations / cross edges / features / K / architecture) |

**Drivers**

| Script | Purpose |
|---|---|
| `run_multiseed.sh` | Multi-seed training driver |
| `run_ksensitivity.sh` | K-sensitivity sweep driver |

---

## Environment

Developed and run with:

- Python 3.10
- PyTorch 2.1.0 (CUDA 12.1 build)
- PyTorch Geometric 2.6.1
- NumPy, pandas, SciPy, scikit-learn
- GeoPandas + Shapely (CBG shapefiles / spatial joins)
- Transformers (BERT POI-text embeddings, `bert-base-uncased`)
- Optuna (hyperparameter search), Matplotlib (figures)

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1
pip install -r requirements.txt
```

Experiments ran on a single NVIDIA H100 GPU. Training the full model for 1000 epochs takes
roughly 2 minutes of wall-clock time with a peak host-memory footprint of about 1.5 GB.

---

## Data

The study uses **SafeGraph POI visit data** (Patterns) and **U.S. Census / ACS** Census Block
Group attributes for the study area (Fulton County and Clarke County, Georgia), with the
corresponding TIGER/Line CBG shapefiles.

These datasets are **not redistributed here** because the POI data is subject to a commercial
license. To reproduce the pipeline, obtain: (1) SafeGraph Patterns for the study weeks (visit
counts, visitor home CBGs, dwell-time buckets, brand, NAICS, open hours); (2) ACS 5-year CBG
tables for the demographic/socioeconomic features; (3) TIGER/Line CBG shapefiles
(e.g. `tl_2019_13_bg`).

Feature engineering (before graph construction) produces, per POI: the BERT POI-text embedding
(reduced to 5 dims), the 7-bucket dwell-time distribution, the visit-derived features, and the
POI–POI edge sets (geographic kNN, week-1 temporal similarity, brand/co-visit); per CBG: the
ACS socioeconomic attributes and engineered accessibility indices. These preprocessing steps
are described in the paper; preprocessing scripts can be provided on request.

> **Leakage-free temporal setting.** For the reported results, node features and all edges are
> computed from **week 1**, while the prediction target is taken from **week 2**, via
> `build_graph_temporal.py` (`--label_poi_csv` supplies the week-2 labels). No feature or edge
> is derived from the target week.

---

## Pipeline

End-to-end: **build graph → split → (optional) tune → train → infer → evaluate.**
Example (leakage-free week-1 → week-2 setting; replace paths with your own):

```bash
# 1) Build the leakage-free graph (week-1 features/edges, week-2 labels)
python build_graph_temporal.py \
    POI_week1.csv  CBG.csv  tl_2019_13_bg.shp  out/graph \
    --run_name fulton_w1f_w2l --k 544 --sample_size 8000 \
    --poi_text_csv poi_text.parquet --plgot_poi_csv poi_anchor.csv \
    --plgot_dwell_npy dwell_feat.npy \
    --plgot_knn_npz knn_edges.npz --plgot_time_npz time_sim_edges.npz \
    --plgot_brand_npz brand_edges.npz --l2_norm_text \
    --label_poi_csv POI_week2.csv --min_visits 10

# 2) Deterministic split
python data_split.py --graph_path out/graph/fulton_w1f_w2l.pt \
    --out_path out/graph/fulton_w1f_w2l_split.pt --seed 42

# 3) (optional) hyperparameter search
python hyper.py --graph_path out/graph/fulton_w1f_w2l_split.pt --out_dir out/hyper

# 4) Train (final tuned configuration)
python train_pp_optimized.py \
    --graph_path out/graph/fulton_w1f_w2l_split.pt --out_dir out/train \
    --epochs 1000 --seed 42 \
    --gate_init 0.12 --gate_anneal cosine --gate_anneal_warmup 80 \
    --learnable_rel_scale --rel_gates \
    --edge_temp_geo 1.2 --edge_temp_time 1.2 \
    --pp_l2 5e-6 --dropout 0.15 --weight_decay 3e-5

# 5) Inference
echo '{"k": 50}' > out/train/params.json
python inferencecopy.py \
    --graph_path out/graph/fulton_w1f_w2l_split.pt \
    --ckpt_path out/train/best_visit_gnn.pt --params_json out/train/params.json \
    --mapping_csv out/graph/fulton_w1f_w2l_poi_to_cbg_mapping.csv \
    --output_dir out/train/prediction --device cuda:0 \
    --force_with_poi_poi --poi_poi_modes geo_knn,time_sim

# 6) Evaluate
python evaluate_and_plot.py \
    --graph_path out/graph/fulton_w1f_w2l_split.pt \
    --preds_csv out/train/prediction/fulton_w1f_w2l_split_preds.csv \
    --out_dir out/train/prediction/eval --split test --ndcg_k 50 --recall_k 5 --match_by auto
```

### Final hyperparameters

1000 epochs; hidden widths d_cbg = 256, d_poi = 128, d_hidden = 64; candidate set K = 50;
gate initialisation 0.12; cosine gate-annealing with 80-epoch warmup; learnable relation
scales and gates; geographic/temporal edge temperatures 1.2; pp_l2 = 5e-6; dropout 0.15;
weight decay 3e-5; learning rate 8.5e-4. Canonical split seed 42; multi-seed runs use seeds
0–4. The full model has ~0.51M trainable parameters.

---

## Reproducing the experiments

**Baselines** — train each baseline, then aggregate into a common-test-POI table:

```bash
python baseline_gbdt.py ...              # gradient-boosted tree
python baseline_hetero_gnn.py ...        # RGCN / HAN (+ distance-augmented)
python baselines_gravity_radiation.py ...# gravity / 2SFCA / radiation
python baseline_table.py --gnn_glob ... --baseline "GBDT=..." --baseline_glob "RGCN=..."
```

**Cross-city** — score Fulton, Athens (retrain), and Fulton→Athens (zero-shot) on each set's
own valid test POIs:

```bash
python crosscity_table.py \
    --entry "Fulton (in-region test)=.../fulton_..._with_gt.csv" \
    --entry_glob "Athens (independent retrain)=.../athens_.../seed_*/.../*_with_gt.csv" \
    --entry "Fulton->Athens (zero-shot)=.../transfer_..._all_with_gt.csv" \
    --ndcg_k 50 --recall_k 5
```

**Candidate coverage & K-sensitivity:**

```bash
python candidate_coverage.py ...
bash run_ksensitivity.sh ...   # then: python ksensitivity_table.py ...  /  plot_ksensitivity.py ...
```

**Ablation matrix** (component and feature ablations, mean±std over seeds, KL on the same scale
as the main results):

```bash
python train_ablation.py \
    --graph_path out/graph/fulton_w1f_w2l_split.pt --out_dir out/ablation \
    --suite poi_poi,cbg_adj,features,feats_plus,arch,cross,K \
    --seeds 41,42,43 --epochs 1000 --k_train 50 --k_eval 50 --ndcg_k 50 --recall_k 5 \
    --feature_variant base+text+dwell \
    --poi_features_json out/graph/fulton_w1_poi_feature_schema.json \
    --cbg_features_json out/graph/fulton_w1_cbg_feature_schema.json
```

The ablation suites are: `poi_poi` (per-relation), `cross` (POI↔CBG edges and edge attributes),
`cbg_adj`, `arch` (GraphNorm), `features` (text/dwell), `feats_plus` (visit-derived /
accessibility / socioeconomic feature groups), and `K` (candidate-set size). Results are
written to `ablation_per_seed.csv` and `ablation_aggregate.csv`.

---

## Citation

If you use this code, please cite the paper (citation to be added upon publication).
