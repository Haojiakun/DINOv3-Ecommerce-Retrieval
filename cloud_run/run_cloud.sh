#!/bin/bash
# Cloud GPU full pipeline for balanced 5K subset (15:85 men:women)
# Upload the entire cloud_run/ directory to the cloud GPU, then:
#   cd cloud_run && bash run_cloud.sh
#
# Outputs go to outputs/ (seeds_local_results.md, gcanet_results.md,
# backbone_results.md, dirty_results.md, visual_results.md)

set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python}
WORK=work
OUT=outputs
mkdir -p "$OUT" "$WORK/features"

echo "=========================================="
echo "[0/12] Check environment"
echo "=========================================="
$PY -c "import torch, timm, transformers, pyarrow, PIL, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=========================================="
echo "[1/12] DINOv3 feature extraction"
echo "=========================================="
$PY $WORK/extract_features.py

echo "=========================================="
echo "[2/12] CLIP text features"
echo "=========================================="
$PY $WORK/build_text.py

echo "=========================================="
echo "[3/12] BLIP-VQA attributes (skip if vqa_attrs.json exists)"
echo "=========================================="
$PY $WORK/extract_attrs_vqa.py

echo "=========================================="
echo "[4/12] Filter attributes + CLIP attr embeddings"
echo "=========================================="
$PY $WORK/filter_attrs.py

echo "=========================================="
echo "[5/12] Train DINOv3 visual head (Table 3)"
echo "=========================================="
$PY $WORK/train_visual.py 2>&1 | tee $OUT/visual_results.md

echo "=========================================="
echo "[6/12] 3-seed cross-modal alignment (Table 2)"
echo "=========================================="
$PY $WORK/run_seeds_local.py

echo "=========================================="
echo "[7/12] GCA-Net baseline (Table 2)"
echo "=========================================="
$PY $WORK/run_gcanet.py

echo "=========================================="
echo "[8/12] Extract DINOv2 features"
echo "=========================================="
TAG=dinov2 MODEL=vit_base_patch14_dinov2.lvd142m $PY $WORK/extract_backbone.py

echo "=========================================="
echo "[9/12] Extract SigLIP features"
echo "=========================================="
TAG=siglip MODEL=vit_base_patch16_siglip_224 $PY $WORK/extract_backbone.py

echo "=========================================="
echo "[10/12] Extract CLIP features"
echo "=========================================="
TAG=clip MODEL=vit_base_patch16_clip_224.openai $PY $WORK/extract_backbone.py

echo "=========================================="
echo "[11/12] Train visual heads + run backbone comparison (Table 4)"
echo "=========================================="
for TAG in dinov2 siglip clip; do
    echo "--- training visual_head_${TAG}.pt ---"
    TAG=$TAG $PY $WORK/train_visual_backbone.py
    echo "--- running alignment for ${TAG} ---"
    TAG=$TAG $PY $WORK/run_backbone_global.py 2>&1 | tee -a $OUT/backbone_results.md
done

echo "=========================================="
echo "[12/12] Dirty data ablation (Table 5)"
echo "=========================================="
$PY $WORK/dirty_ablation.py 2>&1 | tee $OUT/dirty_results.md

echo ""
echo "=========================================="
echo "ALL DONE. Check outputs/ for result files."
echo "=========================================="
