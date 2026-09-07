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
        
        embeddings = self.encode(frames)

        # [B, T, 6] -> [B, T, D]
        action_embeddings = self.action_encoder(actions)

        predictions = self.predictor(embeddings[:, :-1], action_embeddings[:, :-1])

        targets = embeddings[:, 1:]

        return predictions, targets, embeddings

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 4:
            return self.encoder(frames)
        
        if frames.ndim != 5:
            raise ValueError("Supplied game frames tensor is not 4-5 dimensions")

        B, T = frames.shape[:2]
        flat_frames = einops.rearrange(frames, "b t c h w -> (b t) c h w")
        
        flat_embeddings = self.encoder(flat_frames)
        embeddings = einops.rearrange(flat_embeddings, "(b t) d -> b t d", b=B, t=T)
        return embeddings

    @torch.no_grad()
    def rollout(self, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # initial: [B, D]
        # actions: [B, H, 6]
        # result: [B, H, D], where H is the planning horizon. Basically how many future states produced per rollout

        # [B, D] -> [B, 1, D]
        states = initial.unsqueeze(1)

        # Encode all the propsed actions
        conditions = self.action_encoder(actions)

        max_context = self.predictor.pos.size(1)

        for step in range(actions.size(1)):
            context = min(states.size(1), max_context)

            state_context = states[:, -context:]
            action_context = conditions[:, step + 1 - context : step + 1]

            predictions = self.predictor(state_context, action_context)
            next_state = predictions[:, -1:]
            states = torch.cat((states, next_state), dim=1)
        
        # Remove the original state given, returning only the predicted states
        return states[:, 1:]