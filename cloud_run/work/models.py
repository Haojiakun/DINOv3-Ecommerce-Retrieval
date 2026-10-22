import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learn a softmax weight for each patch token and pool them."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, patch_feats):
        # patch_feats: (B, N, embed_dim)
        weights = self.attention(patch_feats)  # (B, N, 1)
        weights = torch.softmax(weights, dim=1)
        pooled_feat = torch.sum(patch_feats * weights, dim=1)  # (B, embed_dim)
        return pooled_feat


class VisualHead(nn.Module):
    """Fuse CLS and attention-pooled patches, then project to a lower dim."""

    def __init__(self, embed_dim=768, output_dim=256):
        super().__init__()
        self.attn_pool = AttentionPooling(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, cls_token, patch_tokens):
        # cls_token: (B, embed_dim), patch_tokens: (B, N, embed_dim)
        pooled = self.attn_pool(patch_tokens)
        fused = torch.cat([cls_token, pooled], dim=-1)
        return self.proj(fused)  # (B, output_dim)


class TextProjection(nn.Module):
    """Project CLIP text features into the same space as VisualHead."""

    def __init__(self, input_dim=512, output_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, text_feats):
        return self.proj(text_feats)  # (B, output_dim)
