import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Splits an image into patches and linearly embeds each one."""

    def __init__(self, img_size=32, patch_size=4, in_ch=3, dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # A conv with stride = patch_size splits and embeds patches in one step
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                       # (B, 3, 32, 32)
        x = self.proj(x)                        # (B, dim, 8, 8)
        return x.flatten(2).transpose(1, 2)     # (B, 64, dim)


class TransformerBlock(nn.Module):
    """One Transformer encoder block: self-attention + MLP, each with a residual."""

    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """A small Vision Transformer for image classification."""

    def __init__(self, img_size=32, patch_size=4, num_classes=10,
                 dim=192, depth=6, heads=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        n = self.patch_embed.num_patches

        # Learnable [CLS] token, prepended to the patch sequence, used for the final prediction
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Learnable positional embeddings (patches have no inherent order otherwise)
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.blocks(x)
        return self.head(self.norm(x)[:, 0])    # predict from the CLS token


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through the model
    model = ViT()
    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print("Output shape:", out.shape)   # expected: torch.Size([2, 10])
