# Defensive baselines and generalization plan

## SCDA baseline (DeepFashion test set, image->image)

SCDA: ImageNet-pretrained ResNet-50, aggregate conv channels, threshold on the
mean response map, sum-pool each channel over selected regions, L2 normalize.
(Simplified SCDA without PCA whitening.)

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| SCDA (ResNet-50) | 28.4 | 48.4 | 58.7 |
| DINOv3 CLS | 34.1 | 54.5 | 62.9 |
| Ours (DINOv3 + hard + all-attribute local) | 79.6 | - | 96.7 |

Traditional pure-visual selective aggregation is far below attribute-guided
local alignment, supporting the core claim.

## Flickr30K generalization experiment (plan)

Goal: verify the alignment framework beyond e-commerce. Fine-grained attributes
are approximated by noun phrases extracted from captions (e.g. "red shirt",
"a dog", "a man").

1. Data
   - Images: `datasets.load_dataset("nlphuji/flickr30k")` (or the original
     Flickr30K images).
   - Captions: Karpathy split `dataset_flickr30k.json` (5 captions per image).

2. Noun-phrase extraction (lightweight NLP)
   ```bash
   python -m pip install spacy
   python -m spacy download en_core_web_sm
   ```
   Extract noun chunks from each caption; use the top chunks as pseudo-attributes.

3. Run the same pipeline
   - Visual: DINOv3 patch features -> VisualHead.
   - Text: CLIP text encoding of caption and of each noun-phrase pseudo-attribute.
   - Loss: symmetric InfoNCE + noun-phrase local patch alignment.
   - Metrics: text->image R@1/R@5/R@10 and image->image R@1 on the 1K test split.

4. Writeup: state that Flickr30K validates general cross-modal alignment, with
   noun phrases standing in for domain attributes. Optionally use Flickr30K-CFQ
   (compact/fragmented queries) for a more fine-grained text->image setup.

## Artifacts

- work/scda_baseline.py, work/features/scda_test.npy, work/resnet50.safetensors.


## Concrete run (Flickr30K 1K test)

```powershell
cd D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper
# download the 141MB test parquet (images + 5 captions/image)
python -c "import urllib.request; urllib.request.urlretrieve('https://hf-mirror.com/datasets/Tristan/flickr30k_test/resolve/main/data/test-00000-of-00001-0ef150b6e3b73917.parquet', 'work/flickr30k_test.parquet')"
# run zero-shot transfer evaluation (CLIP native vs our DeepFashion-trained alignment)
work\venv\Scripts\python.exe work\flickr30k_align.py
```

`work/flickr30k_align.py` computes DINOv3 features, CLIP text/image features,
applies the DeepFashion-trained VisualHead + TextProjection zero-shot, and reports
standard Flickr30K 1K text->image and image->text Recall@1/5/10.
