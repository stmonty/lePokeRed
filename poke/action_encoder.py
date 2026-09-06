import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, n_buttons: int = 6, dim: int = 192, mlp_scale: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_buttons, mlp_scale * dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, 6] -> [B, T, 192], applied independently at every timestep
        return self.net(x)
