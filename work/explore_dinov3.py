import os
import time

# Use a reachable Hugging Face mirror from this network.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Keep the model cache inside the workspace; the home dir is not writable here.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_home"))
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import torch
import timm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} timm={timm.__version__}")

    t0 = time.time()
    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m",
        pretrained=True,
        num_classes=0,  # backbone only, no classifier head
    )
    model = model.to(device).eval()
    print(f"model loaded in {time.time() - t0:.1f}s")

    # Model hyper-parameters.
    print("embed_dim=", model.embed_dim)
    print("patch_size=", model.patch_embed.patch_size)
    print("num_blocks=", len(model.blocks))
    print("num_heads=", model.blocks[0].attn.num_heads)
    print("num_prefix_tokens=", model.num_prefix_tokens)

    # Dummy image: batch=2, RGB 256x256 (DINOv3 pretraining resolution).
    x = torch.randn(2, 3, 256, 256, device=device)

    with torch.inference_mode():
        feats = model.forward_features(x)

    print("--- forward_features output ---")
    if isinstance(feats, dict):
        for k, v in feats.items():
            if torch.is_tensor(v):
                print(f"{k}: {tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"{k}: {type(v)}")
    else:
        print(type(feats), tuple(feats.shape))

    # Token layout for DINOv3: [CLS, reg x num_reg_tokens, patch x H*W].
    n_prefix = model.num_prefix_tokens
    cls = feats[:, 0]
    reg = feats[:, 1:n_prefix] if n_prefix > 1 else None
    patches = feats[:, n_prefix:]
    print("cls token:", tuple(cls.shape))
    if reg is not None:
        print(f"register tokens (x{n_prefix - 1}):", tuple(reg.shape))
    print("patch tokens:", tuple(patches.shape))

    if torch.cuda.is_available():
        print(f"gpu mem allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
