"""Models: the VGG-style CNN baseline, ResNet-18, a small ViT, and the MAE
wrapper used for masked pre-training."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
class VGGStyleCNN(nn.Module):
    """PyTorch port of the original Keras baseline: three conv blocks
    (32, 64, 128 filters, 3x3, ReLU, BatchNorm, 2x2 max-pool), then
    Dense(128) + Dropout(0.5) + Dense(8)."""

    def __init__(self, n_classes=8):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3), nn.ReLU(inplace=True),
                                 nn.BatchNorm2d(cout), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 6, 128),
                                  nn.ReLU(inplace=True), nn.Dropout(0.5),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18(n_classes=8):
    """torchvision ResNet-18, trained from scratch, with the stem shrunk for
    64x64 inputs (3x3 stride-2 conv instead of 7x7 stride-2)."""
    m = torchvision.models.resnet18(weights=None, num_classes=n_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
    return m


# ----------------------------------------------------------------------
# Vision Transformer
# ----------------------------------------------------------------------
def sincos_2d(dim, grid):
    """Fixed 2-D sine-cosine position embedding, (grid*grid, dim)."""
    def one_d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2))
        out = pos.reshape(-1)[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                            torch.arange(grid, dtype=torch.float32), indexing="ij")
    return torch.cat([one_d(dim // 2, ys), one_d(dim // 2, xs)], dim=1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)      # softmax(QK^T / sqrt(d_k)) V
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    """Small ViT encoder. `forward_tokens` accepts an optional index of patches
    to keep, which is how the MAE feeds it only the visible ones."""

    def __init__(self, img=64, patch=8, dim=128, depth=4, heads=2, n_classes=8):
        super().__init__()
        self.patch, self.grid, self.dim = patch, img // patch, dim
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", sincos_2d(dim, self.grid)[None], persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.apply(_init)

    def forward_tokens(self, x, keep=None):
        x = self.embed(x).flatten(2).transpose(1, 2) + self.pos      # (B, N, D)
        if keep is not None:                                          # (B, n_keep)
            x = torch.gather(x, 1, keep[..., None].expand(-1, -1, x.size(2)))
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                                           # (B, 1+n, D)

    def features(self, x):
        """Image -> one vector: the mean of the patch tokens."""
        return self.forward_tokens(x)[:, 1:].mean(1)

    def forward(self, x):
        return self.head(self.features(x))


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------
# Masked autoencoder
# ----------------------------------------------------------------------
class MAE(nn.Module):
    """Masked autoencoder (He et al., 2022). The encoder sees only the visible
    patches; a light decoder reconstructs the pixels of the hidden ones.

    The only thing that changes between experiments is *which* patches are
    hidden: the caller passes `keep` and `masked` indices from masking.py.
    """

    def __init__(self, encoder: ViT, dec_dim=64, dec_depth=1, dec_heads=2):
        super().__init__()
        self.encoder = encoder
        n = encoder.grid ** 2
        self.to_dec = nn.Linear(encoder.dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_2d(dec_dim, encoder.grid)[None], persistent=False)
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.to_pixels = nn.Linear(dec_dim, encoder.patch ** 2 * 3)
        self.n = n
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in [self.to_dec, self.dec_blocks, self.dec_norm, self.to_pixels]:
            m.apply(_init)

    def patchify(self, x):
        p = self.encoder.patch
        B, C, H, W = x.shape
        x = x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, (H // p) * (W // p), p * p * C)

    def forward(self, x, keep, masked):
        B = x.size(0)
        z = self.to_dec(self.encoder.forward_tokens(x, keep)[:, 1:])     # visible tokens
        full = self.mask_token.expand(B, self.n, -1).clone()
        full.scatter_(1, keep[..., None].expand(-1, -1, z.size(2)), z)   # put them back in place
        full = full + self.dec_pos
        for blk in self.dec_blocks:
            full = blk(full)
        pred = self.to_pixels(self.dec_norm(full))                       # (B, N, p*p*3)

        target = self.patchify(x)
        # per-patch normalised pixels: predict structure, not brightness
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        loss = ((pred - target) ** 2).mean(-1)                           # (B, N)
        return torch.gather(loss, 1, masked).mean()
