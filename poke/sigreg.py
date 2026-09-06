import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, num_projections: int = 1024, num_knots: int = 17) -> None:
        super().__init__()
        self.num_projections = num_projections

        t = torch.linspace(0, 3, num_knots)
        phi = torch.exp(-(t ** 2) / 2)
        dt = 3 / (num_knots - 1)
        integration_weights = torch.full((num_knots,), 2 * dt, dtype=t.dtype)
        integration_weights[0] = dt
        integration_weights[-1] = dt
        weights = integration_weights * phi
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:

        T, B, D = embeddings.shape
        P = torch.nn.functional.normalize(torch.randn(
                        self.num_projections, D, device=embeddings.device, dtype=embeddings.dtype), dim=-1)
        shadows = embeddings @ P.transpose(0, 1)
        assert shadows.shape == (T, B, self.num_projections)
        
        angles = shadows.unsqueeze(dim=-1) * self.t
        
        cos_average = angles.cos().mean(dim=1)
        sin_average = angles.sin().mean(dim=1)
        error = (cos_average - self.phi).square() + sin_average.square()
        statistic = (error @ self.weights) * embeddings.size(dim=1)
        return statistic.mean()
