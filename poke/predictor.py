import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def causal_mask(t: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        # affine off: adaLN supplies the scale and shift instead
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shift_attn, scale_attn)
        mask = causal_mask(x.size(1), x.device)
        attended = self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + gate_attn * attended

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)
        return x


class Predictor(nn.Module):
    def __init__(
        self,
        num_frames: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ConditionalBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        x = self.drop(emb + self.pos[:, : emb.size(1)])

        for block in self.blocks:
            x = block(x, act_emb)

        return self.norm(x)
