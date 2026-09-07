import torch as t
import einops
from poke.sigreg import SIGReg
from poke.jepa import JEPA
from poke.dataset import PokeDataset
from poke.splits import episode_split
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass

@dataclass
class DefaultConfig:
    data_path: str = "episodes/poke.h5"
    num_frames: int = 4
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    regularizer_weight: float = 0.1
    seed: int = 42

def loss_terms(model: JEPA, reg: SIGReg, batch: dict[str, t.Tensor], regularizer_weight: float = 0.1) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    predictions, targets, embeddings = model(batch["frames"], batch["actions"])
    prediction_loss = t.nn.functional.mse_loss(predictions, targets)
    temporal_embeddings = einops.rearrange(embeddings, "b t d -> t b d")
    reg_loss = reg(temporal_embeddings)

    # This represents: L_pred + λL_reg (and yes I had to find a lambda character on Google)
    total_loss = prediction_loss + (regularizer_weight * reg_loss)
    
    return total_loss, prediction_loss, reg_loss


def optimization_step(model: JEPA, reg: SIGReg, optimizer: Optimizer, batch: dict[str, t.Tensor], regularizer_weight: float = 0.1) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    model.train()

    optimizer.zero_grad(set_to_none=True)
    total_loss, prediction_loss, reg_loss = loss_terms(model, reg, batch, regularizer_weight)

    # Calculate gradients
    total_loss.backward()

    optimizer.step()

    return (total_loss.detach(), prediction_loss.detach(), reg_loss.detach())

def make_loaders(data_path: str, num_frames: int = 4, batch_size: int = 16, num_workers: int = 0, val_fraction: float = 0.1, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    train_episodes, val_episodes = episode_split(data_path, val_fraction=val_fraction, seed=seed)

    train_dataset = PokeDataset(data_path, num_frames=num_frames, episode_indices=train_episodes)
    val_dataset = PokeDataset(data_path, num_frames=num_frames, episode_indices=val_episodes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers, pin_memory=t.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,num_workers=num_workers, pin_memory=t.cuda.is_available())

    return (train_loader, val_loader)


@t.no_grad()
def validate(model: JEPA, reg: SIGReg, loader: DataLoader, device: t.device, regularizer_weight: float = 0.1) -> tuple[float, float, float]:
    model.eval()
    loss_sums = t.zeros(3, device=device)
    number_of_samples = 0

    for batch in tqdm(loader, desc="validation", unit="batch", leave=False):
        batch = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()
        }

        losses = loss_terms(model, reg, batch, regularizer_weight)
        batch_size = batch["frames"].size(0)
        loss_sums += t.stack(losses) * batch_size
        number_of_samples += batch_size

    averages = loss_sums / number_of_samples
    return tuple(averages.cpu().tolist())

def train_one_epoch(
    model: JEPA,
    reg: SIGReg,
    optimizer: Optimizer,
    loader: DataLoader,
    device: t.device,
    regularizer_weight: float = 0.1,
) -> tuple[float, float, float]:
    model.train()

    loss_sums = t.zeros(3, device=device)
    number_of_samples = 0

    progress = tqdm(loader, desc="training", unit="batch")

    for batch_index, batch in enumerate(progress, start=1):
        batch = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()
        }

        losses = optimization_step(model, reg, optimizer, batch, regularizer_weight)

        batch_size = batch["frames"].size(0)

        loss_sums += t.stack(losses) * batch_size
        number_of_samples += batch_size

        if batch_index % 10 == 0 or batch_index == len(loader):
            running_averages = loss_sums / number_of_samples
            progress.set_postfix(
                total=f"{running_averages[0].item():.4f}",
                pred=f"{running_averages[1].item():.4f}",
                reg=f"{running_averages[2].item():.4f}",
            )

    averages = loss_sums / number_of_samples
    return tuple(averages.cpu().tolist())

def main() -> None:
    config = DefaultConfig()
    t.manual_seed(config.seed)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(data_path=config.data_path, num_frames=config.num_frames, batch_size=config.batch_size, seed=config.seed)

    model = JEPA(num_frames=config.num_frames).to(device)
    reg = SIGReg().to(device)

    optimizer = t.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"device: {device}")
    print(f"train windows: {len(train_loader.dataset)}")
    print(f"validation windows: {len(val_loader.dataset)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config.epochs):
        train_losses = train_one_epoch(
            model,
            reg,
            optimizer,
            train_loader,
            device,
            config.regularizer_weight,
        )

        val_losses = validate(
            model,
            reg,
            val_loader,
            device,
            config.regularizer_weight,
        )

        print(
            f"epoch {epoch + 1:02d} | "
            f"train total={train_losses[0]:.4f} "
            f"pred={train_losses[1]:.4f} "
            f"reg={train_losses[2]:.4f} | "
            f"val total={val_losses[0]:.4f} "
            f"pred={val_losses[1]:.4f} "
            f"reg={val_losses[2]:.4f}"
        )

if __name__ == "__main__":
    main()
