import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        height: int = 144,
        width: int = 160,
        patch: int = 16,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        mlp_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        n_patches = (height // patch) * (width // patch)

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)

        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1, 144, 160] -> [B, 90, 192]
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos

        for block in self.blocks:
            x = block(x)

        return self.norm(x)[:, 0]
