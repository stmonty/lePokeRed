"""Fine-tune JEPA's dynamics on recursive, multi-step latent rollouts."""

import argparse
from pathlib import Path

import torch as t
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from poke.dataset import PokeDataset
from poke.jepa import JEPA
from poke.splits import episode_split


def batches_to_process(loader: DataLoader, max_batches: int | None) -> int:
    """Validate a batch limit and return the number of batches to process."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")

    available_batches = len(loader)
    if available_batches == 0:
        raise ValueError(
            "data loader has no batches; check the dataset split and batch size"
        )

    if max_batches is None:
        return available_batches
    return min(available_batches, max_batches)


def make_loaders(
    data_path: str,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_episodes, val_episodes = episode_split(data_path, seed=seed)
    train_dataset = PokeDataset(
        data_path,
        num_frames=num_frames,
        episode_indices=train_episodes,
    )
    val_dataset = PokeDataset(
        data_path,
        num_frames=num_frames,
        episode_indices=val_episodes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=t.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=t.cuda.is_available(),
    )
    return train_loader, val_loader


def curriculum_horizon(epoch: int, epochs: int, max_horizon: int) -> int:
    """Progress through 3-, 6-, and full-horizon recursive training."""
    if epochs < 1 or max_horizon < 1:
        raise ValueError("epochs and max_horizon must be positive")
    if not 0 <= epoch < epochs:
        raise ValueError("epoch must be in the range [0, epochs)")

    fraction_complete = (epoch + 1) / epochs
    if fraction_complete <= 1 / 3:
        return min(3, max_horizon)
    if fraction_complete <= 2 / 3:
        return min(6, max_horizon)
    return max_horizon


def rollout_loss_terms(
    model: JEPA,
    embeddings: t.Tensor,
    actions: t.Tensor,
    horizon: int,
    teacher_weight: float,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Return combined, recursive-rollout, and local one-step losses."""
    if horizon < 1:
        raise ValueError("horizon must be positive")

    targets = embeddings[:, 1 : horizon + 1].detach()
    transition_actions = actions[:, :horizon]

    rollout_predictions = model.predict_rollout(
        embeddings[:, 0].detach(),
        transition_actions,
    )
    rollout_loss = F.mse_loss(rollout_predictions, targets)

    batch_size, _, dimension = targets.shape
    real_states = embeddings[:, :horizon].detach().reshape(
        batch_size * horizon,
        1,
        dimension,
    )
    real_actions = transition_actions.reshape(batch_size * horizon, 1, -1)
    one_step_predictions = model.predictor(
        real_states,
        model.action_encoder(real_actions),
    ).reshape(batch_size, horizon, dimension)
    teacher_loss = F.mse_loss(one_step_predictions, targets)

    total_loss = rollout_loss + teacher_weight * teacher_loss
    return total_loss, rollout_loss, teacher_loss


def train_one_epoch(
    model: JEPA,
    optimizer: Optimizer,
    loader: DataLoader,
    device: t.device,
    horizon: int,
    teacher_weight: float,
    max_batches: int | None,
) -> tuple[float, float, float]:
    total_batches = batches_to_process(loader, max_batches)
    model.train()
    model.encoder.eval()
    loss_sums = t.zeros(3, device=device)
    samples_seen = 0

    progress = tqdm(loader, total=total_batches, desc=f"rollout h={horizon}")

    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)

        with t.no_grad():
            embeddings = model.encode(frames)

        losses = rollout_loss_terms(
            model,
            embeddings,
            actions,
            horizon,
            teacher_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses[0].backward()
        t.nn.utils.clip_grad_norm_(
            [
                *model.predictor.parameters(),
                *model.action_encoder.parameters(),
            ],
            max_norm=1.0,
        )
        optimizer.step()

        batch_size = frames.size(0)
        loss_sums += t.stack([loss.detach() for loss in losses]) * batch_size
        samples_seen += batch_size

        if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
            averages = loss_sums / samples_seen
            progress.set_postfix(
                total=f"{averages[0].item():.4f}",
                rollout=f"{averages[1].item():.4f}",
                one_step=f"{averages[2].item():.4f}",
            )

    if samples_seen == 0:
        raise RuntimeError("training processed no samples")
    averages = loss_sums / samples_seen
    return tuple(averages.cpu().tolist())


@t.inference_mode()
def validate_rollout(
    model: JEPA,
    loader: DataLoader,
    device: t.device,
    max_horizon: int,
    max_batches: int | None,
) -> list[float]:
    """Return held-out recursive MSE separately for every future step."""
    if max_horizon < 1:
        raise ValueError("max_horizon must be positive")

    total_batches = batches_to_process(loader, max_batches)
    model.eval()
    squared_error_sums = t.zeros(max_horizon, device=device)
    values_per_horizon = 0

    progress = tqdm(loader, total=total_batches, desc="validate rollout", leave=False)

    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        embeddings = model.encode(frames)
        predictions = model.rollout(
            embeddings[:, 0],
            actions[:, :max_horizon],
        )
        targets = embeddings[:, 1 : max_horizon + 1]

        squared_error_sums += (predictions - targets).square().sum(dim=(0, 2))
        values_per_horizon += frames.size(0) * embeddings.size(-1)

    if values_per_horizon == 0:
        raise RuntimeError("validation processed no values")
    return (squared_error_sums / values_per_horizon).cpu().tolist()


def save_checkpoint(
    path: Path,
    model: JEPA,
    optimizer: Optimizer,
    source_checkpoint: str,
    source_config: dict,
    epoch: int,
    global_step: int,
    train_losses: tuple[float, float, float],
    validation_horizon_mse: list[float],
    rollout_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    t.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "config": source_config,
            "rollout_config": rollout_config,
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "jepa": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "validation_horizon_mse": validation_horizon_mse,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a frozen JEPA encoder for stable latent rollouts"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="episodes/poke.h5")
    parser.add_argument("--output-dir", default="checkpoints/rollout")
    parser.add_argument("--max-horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if t.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.max_horizon < 1:
        raise ValueError("max-horizon must be positive")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.teacher_weight < 0:
        raise ValueError("teacher-weight must be non-negative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max-train-batches must be positive when provided")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        raise ValueError("max-val-batches must be positive when provided")

    t.manual_seed(args.seed)
    device = t.device(args.device)
    source = t.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = JEPA(num_frames=source["config"]["num_frames"]).to(device)
    model.load_state_dict(source["jepa"])

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)

    trainable_parameters = [
        *model.predictor.parameters(),
        *model.action_encoder.parameters(),
    ]
    optimizer = t.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader, val_loader = make_loaders(
        args.data,
        num_frames=args.max_horizon + 1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    rollout_config = {
        "data": str(Path(args.data).resolve()),
        "max_horizon": args.max_horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "teacher_weight": args.teacher_weight,
        "seed": args.seed,
    }
    output_dir = Path(args.output_dir)
    best_mean_horizon_mse = float("inf")
    global_step = 0

    print(f"device: {device}")
    print(f"train windows: {len(train_loader.dataset)}")
    print(f"validation windows: {len(val_loader.dataset)}")
    print(f"frozen encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"trainable dynamics parameters: {sum(p.numel() for p in trainable_parameters):,}")

    try:
        baseline_mse = validate_rollout(
            model,
            val_loader,
            device,
            args.max_horizon,
            args.max_val_batches,
        )
        baseline_summary = " ".join(
            f"h{step}={baseline_mse[step - 1]:.6f}"
            for step in sorted(
                {
                    1,
                    min(3, args.max_horizon),
                    min(6, args.max_horizon),
                    args.max_horizon,
                }
            )
        )
        print(
            f"baseline rollout_mean={sum(baseline_mse) / len(baseline_mse):.6f} "
            f"{baseline_summary}"
        )

        for epoch in range(args.epochs):
            horizon = curriculum_horizon(epoch, args.epochs, args.max_horizon)
            train_losses = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                horizon,
                args.teacher_weight,
                args.max_train_batches,
            )
            validation_mse = validate_rollout(
                model,
                val_loader,
                device,
                args.max_horizon,
                args.max_val_batches,
            )

            batches_this_epoch = len(train_loader)
            if args.max_train_batches is not None:
                batches_this_epoch = min(
                    batches_this_epoch,
                    args.max_train_batches,
                )
            global_step += batches_this_epoch

            reported_horizons = sorted(
                {1, min(3, args.max_horizon), min(6, args.max_horizon), args.max_horizon}
            )
            horizon_summary = " ".join(
                f"h{step}={validation_mse[step - 1]:.6f}"
                for step in reported_horizons
            )
            print(
                f"epoch {epoch + 1:02d} train_h={horizon} "
                f"train_rollout={train_losses[1]:.6f} "
                f"train_one_step={train_losses[2]:.6f} "
                f"val_mean={sum(validation_mse) / len(validation_mse):.6f} "
                f"val {horizon_summary}"
            )

            checkpoint_arguments = (
                model,
                optimizer,
                args.checkpoint,
                source["config"],
                epoch + 1,
                global_step,
                train_losses,
                validation_mse,
                rollout_config,
            )
            save_checkpoint(output_dir / "last.pt", *checkpoint_arguments)

            mean_horizon_mse = sum(validation_mse) / len(validation_mse)
            if mean_horizon_mse < best_mean_horizon_mse:
                best_mean_horizon_mse = mean_horizon_mse
                save_checkpoint(output_dir / "best.pt", *checkpoint_arguments)
                print(
                    f"saved new best rollout checkpoint: "
                    f"{output_dir / 'best.pt'}"
                )
    finally:
        train_loader.dataset.close()
        val_loader.dataset.close()


if __name__ == "__main__":
    main()
