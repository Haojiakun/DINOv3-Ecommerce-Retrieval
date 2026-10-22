# DINOv3 E-Commerce Retrieval

Attribute-guided visual-text alignment for fashion image retrieval, built on
frozen DINOv3 features with a lightweight attention-pooling head and symmetric
InfoNCE loss.

## Results

All numbers are image-to-image (i2i) Recall@K on the held-out test split,
evaluated with the 6:2:2 product-level protocol.

| Scale | #Images | i2i R@1 | i2i R@10 |
|---:|---:|---:|---:|
| Balanced subset | 5,003 | **92.9** | 98.8 |
| Full dataset | 52,591 | **70.1** | 90.1 |

Text-to-image (t2i) on the 5,003-image subset (70 epochs, 3 seeds):

| Metric | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| t2i | 25.7 | 42.0 | 53.0 |

## Method

1. **Frozen DINOv3 backbone** — extract CLS token and 256 patch tokens per image.
2. **VisualHead** — attention-pool patch tokens, fuse with CLS, project to 256-d.
3. **TextProjection** — map CLIP text embeddings to the same 256-d space.
4. **Symmetric InfoNCE** — align visual and text representations; optional
   hard-negative weighting (w=3) and all-attribute local patch alignment.

## Installation

```bash
# Clone the repository
git clone https://github.com/Haojiakun/DINOv3-Ecommerce-Retrieval.git
cd DINOv3-Ecommerce-Retrieval

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data

The experiments use the DeepFashion In-shop dataset. Features are extracted with
`extract_features.py` and cached as `.npy` files (not included in this repo due
to size).

## Usage

Feature extraction:

```bash
python extract_features.py
```

Training (plain InfoNCE, full dataset):

```bash
N=52591 EPOCHS=70 SEEDS=3 python run_cloud.py
```

Training on the 5,003-image balanced subset:

```bash
FEAT_DIR=features_5k N=5003 EPOCHS=70 SEEDS=3 python run_cloud.py
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `N` | 5000 | Number of images to use |
| `EPOCHS` | 50 | Training epochs |
| `SEEDS` | 3 | Number of random seeds |
| `LR` | 1e-3 | Learning rate |
| `TEMP` | 0.07 | InfoNCE temperature |
| `FEAT_DIR` | features | Feature directory |

## Project Structure

```
├── cloud_run/work/          # Cloud training & evaluation scripts
│   ├── run_cloud.py         # Main InfoNCE training (t2i + i2i)
│   ├── models.py            # VisualHead, TextProjection, AttentionPooling
│   ├── train_attr_align.py  # Attribute alignment losses
│   ├── dirty_ablation.py    # Hard-negative & local-patch ablations
│   └── extract_features.py  # DINOv3 feature extraction
├── work/                    # Local scripts (baselines, SCDA, visualization)
├── outputs/                 # Result tables & figures
├── requirements.txt
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{dinov3-ecommerce-retrieval,
  author = {Haojiakun},
  title  = {DINOv3 E-Commerce Retrieval},
  year   = {2026},
  url    = {https://github.com/Haojiakun/DINOv3-Ecommerce-Retrieval}
}
```
