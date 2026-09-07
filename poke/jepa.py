import torch
import torch.nn as nn
import einops
from poke.action_encoder import ActionEncoder
from poke.encoder import Encoder
from poke.predictor import Predictor


class JEPA(nn.Module):
    def __init__(self, num_frames: int = 4, dim: int = 192) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.dim = dim
        self.encoder = Encoder(dim=dim)
        self.predictor = Predictor(num_frames=num_frames - 1, dim=dim)
        self.action_encoder = ActionEncoder(dim=dim)
        

    def forward(self, frames: torch.Tensor, actions: torch.Tensor):
        # frames: [B, T, 1, H, W]
        # actions: [B, T, 6]
        B, T = frames.shape[:2]
        
        # [B, T, 1, H, W] -> [B*T, 1, H, W]
        flat_frames = einops.rearrange(frames, "b t c h w -> (b t) c h w")
        
        # [B*T, 1, H, W] -> [B*T, D]
        flat_embeddings = self.encoder(flat_frames)

        # [B*T, D] -> [B, T, D]
        embeddings = einops.rearrange(flat_embeddings, "(b t) d -> b t d", b=B, t=T)

        # [B, T, 6] -> [B, T, D]
        action_embeddings = self.action_encoder(actions)

        predictions = self.predictor(embeddings[:, :-1], action_embeddings[:, :-1])

        targets = embeddings[:, 1:]

        return predictions, targets, embeddings
