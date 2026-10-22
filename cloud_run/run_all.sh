#!/bin/bash
# Run 5K subset then full 52K dataset on cloud GPU
# batch-level InfoNCE, batch_size=256, 3 seeds, skip VQA, no checkpoint
set -e
cd /root/cloud_run/work

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

echo "========================================"
echo "STEP 1: 5K balanced subset"
echo "========================================"

# Extract DINOv3 features for 5K subset
echo "[1/4] extract_features (5K subset)..."
PARQUET=deepfashion_subset.parquet python3 extract_features.py 2>&1 | tail -20

# Build CLIP text features
echo "[2/4] build_text (5K subset)..."
PARQUET=deepfashion_subset.parquet python3 build_text.py 2>&1 | tail -10

# Run training + eval (3 seeds, batch_size=256, batch-level InfoNCE)
echo "[3/4] run_cloud (5K, 3 seeds)..."
N=5000 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py 2>&1

# Save 5K results
cp ../outputs/cloud_results_N5000*.md /root/results_5k.md 2>/dev/null || true

echo "========================================"
echo "STEP 2: Full 52K dataset"
echo "========================================"

# Extract DINOv3 features for full dataset
echo "[1/3] extract_features (full 52K)..."
PARQUET=deepfashion_inshop.parquet python3 extract_features.py 2>&1 | tail -20

# Build CLIP text features for full dataset
echo "[2/3] build_text (full 52K)..."
PARQUET=deepfashion_inshop.parquet python3 build_text.py 2>&1 | tail -10

# Run training + eval (3 seeds, batch_size=256)
echo "[3/3] run_cloud (full 52K, 3 seeds)..."
N=52591 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py 2>&1

# Save full results
cp ../outputs/cloud_results_N52591*.md /root/results_full.md 2>/dev/null || true

echo "========================================"
echo "ALL DONE"
echo "========================================"
echo "5K results: /root/results_5k.md"
echo "Full results: /root/results_full.md"
