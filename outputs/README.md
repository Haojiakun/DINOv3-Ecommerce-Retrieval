# DINOv3 + DeepFashion first step

## 位置

项目已迁移到 `D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper`。
原 `C:\Users\ASUS\Documents\Codex\...` 下的旧目录只剩一个空壳，关闭 Codex 后运行
`outputs\migrate_codex_settings.ps1` 即可删除旧目录并把 Codex 里的路径改到 D 盘。

## 环境

- 显存: RTX 4060 Laptop 8GB, CUDA 12.5, torch 2.4.1+cu121
- Python: 3.9.1
- 运行环境: `work/venv`(继承系统 torch, 额外装了 `timm 1.0.30` 和 `torchvision 0.19.1`)
- 模型缓存: `work/hf_home`
- 数据: `work/deepfashion_inshop.parquet`(来自 `Marqo/deepfashion-inshop`, 52591 条)

## 运行

```powershell
work\venv\Scripts\python.exe outputs\run_dinov3.py
```

## 关键结论

- 模型: `vit_base_patch16_dinov3.lvd1689m`
- 预训练输入分辨率: 256x256(不是 224), patch 网格 16x16
- `forward_features` 输出 `(B, 261, 768)`:
  - `[:, 0]` -> CLS token
  - `[:, 1:5]` -> 4 个 register token
  - `[:, 5:]` -> 256 个 patch token, 形状 `(B, 256, 768)`
- 做 Attention Pooling 时, patch 输入应取 `feats[:, 5:]`
- CLS 余弦相似度的 sanity check 显示同款不同视角略高于跨款, 但差距不大,
  这正是需要属性引导对齐来拉开 margin 的动机

## Task 1 基线

- `work/extract_features.py`: 提取 5000 张图的 DINOv3 CLS 与 patch 特征到
  `work/features/{cls,patch}.npy`。
- `work/eval_baseline.py`: CLS 图像检索 Recall@K。
- `work/eval_clip.py`: CLIP 原生与“DINOv3 CLS 拼接 CLIP 文本”基线。
- `work/tsne_vis.py`: 生成 CLS 的 t-SNE 图。
- 结果汇总见 `baseline_results.md`。

## Task 2 视觉头

- `work/train_attn.py`: 冻结 DINOv3，训练 AttentionPool 与
  `CLS + AttentionPool + MLP` 两个轻量头，做同款度量学习并验证 Recall@K。
- 权重: `work/features/attn_pool.pt`、`work/features/attn_fusion.pt`。
- 结果汇总见 `attn_results.md`。

## Task 3 文本对齐

- `work/extract_attr_text.py`: 用 CLIP 文本编码器把 `category1/2/3 + color`
  编码成 `work/features/attr_text.npy`。
- `work/train_align.py`: 视觉 FusionHead + 文本投影做 InfoNCE 对齐，并加
  “同品类异材质”困难负样本 margin。
- 权重: `work/features/align_visual.pt`、`align_text_proj.pt`、`align_attr_proj.pt`。
- 结果汇总见 `align_results.md`。

## Task A 视觉模块（正式版）

- `work/models.py`: `AttentionPooling` 与 `VisualHead`。
- `work/train_visual.py`: 纯视觉 InfoNCE 训练。
- `work/evaluate.py`: Recall@K 评测。
- 权重: `work/features/visual_head.pt`。
- 结果汇总见 `visual_results.md`。

## 注意

- 本机直连 `huggingface.co` 超时, 已通过 `HF_ENDPOINT=https://hf-mirror.com` 下载
- 模型缓存和数据集都放在 `work/` 下, 主目录不可写

## Task B 文本对齐（正式版）

- work/build_text.py: item_ID+属性构造自然语言并用 CLIP 编码。
- work/train_align.py: 普通 InfoNCE 与类目加权困难负样本 InfoNCE 对照。
- 权重: work/features/align_visual_{plain,hard}.pt、lign_text_proj_{plain,hard}.pt。
- 结果汇总见 lign_b_results.md。

## 细粒度属性

- work/build_attribute_dict.py: 从 item_ID/category2 构建属性字典。
- work/extract_attrs_vqa.py: BLIP-VQA 抽取材质/领口/袖长/图案。
- work/features/attributes.json、work/features/vqa_attrs.json。
- 说明见 qa_results.md。

## ??????????

- `work/filter_attrs.py`: ?????? VQA ???
- `work/train_attr_align.py`: ????????? + Patch ?????
- ??? `attr_align_results.md`?

## ??

- `work/extract_backbone.py` / `work/eval_backbone.py`: ?????DINOv3/CLIP-ViT/SigLIP??
- `work/dirty_ablation.py`: ??? vs ???????
- ??? `ablation_results.md`?

## ???????

- `work/scda_baseline.py`: SCDA ??????????
- Flickr30K ??????? `scda_flickr.md`?
