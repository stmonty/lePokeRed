# Trying to test if we actually are training the model on selecting the starter pokemon
# After running scripts/train.py
# uv run python scripts/probe.py checkpoints/best.pt

import argparse
import sys

import h5py
import numpy as np
import torch as t
import torch.nn as nn
from tqdm.auto import tqdm

from poke.jepa import JEPA
from poke.splits import episode_split


def frame_indices(
    data_path: str,
    episode_indices: np.ndarray,
) -> np.ndarray:

    with h5py.File(data_path, "r") as file:
        # Each row contains [starting_frame, episode_length].
        episode_bounds = file["episodes"][episode_indices]

    return np.concatenate(
        [
            np.arange(int(start), int(start + length), dtype=np.int64)
            for start, length in episode_bounds
        ]
    )


def balanced_sample(
    data_path: str,
    indices: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:

    with h5py.File(data_path, "r") as file:
        labels = file["party_counts"][indices] > 0

    negative_indices = indices[~labels]
    positive_indices = indices[labels]

    count = min(len(negative_indices), len(positive_indices), max_per_class)
    if count == 0:
        raise RuntimeError("probe split needs both positive and negative frames")

    selected = np.concatenate(
        [
            rng.choice(negative_indices, count, replace=False),
            rng.choice(positive_indices, count, replace=False),
        ]
    )

    # h5py requires fancy indices to be in increasing order.
    return np.sort(selected)


@t.inference_mode()
def encode_frames(
    model: JEPA,
    data_path: str,
    indices: np.ndarray,
    batch_size: int,
    device: t.device,
    description: str,
) -> tuple[t.Tensor, t.Tensor]:

    model.eval()
    feature_batches = []
    label_batches = []

    with h5py.File(data_path, "r") as file:
        starts = range(0, len(indices), batch_size)
        for start in tqdm(starts, desc=description, unit="batch", leave=False):
            selected = indices[start : start + batch_size]

            pixels = (
                file["frames"][selected, :, :, 0].astype(np.float32) / 127.5
                - 1.0
            )
            frames = t.from_numpy(pixels).unsqueeze(1).to(device)

            # The probe uses only the visual encoder.
            features = model.encoder(frames)

            labels = t.from_numpy(
                (file["party_counts"][selected] > 0).astype(np.float32)
            )

            # Keep accumulated features on CPU to conserve GPU memory.
            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())

    return t.cat(feature_batches), t.cat(label_batches)


def roc_auc(scores: t.Tensor, labels: t.Tensor) -> float:
    """Measure ranking quality independently of one decision threshold."""

    order = scores.argsort(descending=True)
    sorted_labels = labels[order].float()

    true_positives = sorted_labels.cumsum(dim=0)
    false_positives = (1.0 - sorted_labels).cumsum(dim=0)

    zero = t.zeros(1, dtype=scores.dtype, device=scores.device)
    true_positive_rate = t.cat((zero, true_positives / true_positives[-1]))
    false_positive_rate = t.cat((zero, false_positives / false_positives[-1]))

    return t.trapezoid(true_positive_rate, false_positive_rate).item()


def train_probe(
    x_train: t.Tensor,
    y_train: t.Tensor,
    epochs: int,
    learning_rate: float,
) -> nn.Linear:

    probe = nn.Linear(x_train.size(1), 1).to(x_train.device)
    optimizer = t.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(epochs):
        logits = probe(x_train).squeeze(1)
        loss = criterion(logits, y_train)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linear probe for party acquisition"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="episodes/poke.h5")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-per-class", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if t.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = t.device(args.device)
    rng = np.random.default_rng(args.seed)
    t.manual_seed(args.seed)

    # Load the trained representation.
    checkpoint = t.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = JEPA(
        num_frames=checkpoint["config"]["num_frames"]
    ).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    # Make it explicit that the JEPA itself must not learn during probing.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_episodes, val_episodes = episode_split(
        args.data,
        seed=args.seed,
    )

    train_indices = balanced_sample(
        args.data,
        frame_indices(args.data, train_episodes),
        args.max_per_class,
        rng,
    )
    val_indices = balanced_sample(
        args.data,
        frame_indices(args.data, val_episodes),
        args.max_per_class,
        rng,
    )

    x_train, y_train = encode_frames(
        model,
        args.data,
        train_indices,
        args.batch_size,
        device,
        "encoding train",
    )
    x_val, y_val = encode_frames(
        model,
        args.data,
        val_indices,
        args.batch_size,
        device,
        "encoding validation",
    )

    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    probe = train_probe(
        x_train,
        y_train,
        args.epochs,
        args.lr,
    )

    with t.inference_mode():
        probe.eval()
        scores = probe(x_val).squeeze(1)
        predictions = scores >= 0
        accuracy = (predictions == y_val.bool()).float().mean().item()
        auc = roc_auc(scores, y_val)

    passed = auc >= args.threshold
    verdict = "GO" if passed else "NO-GO"

    print(
        f"{verdict}: "
        f"auc={auc:.3f} "
        f"accuracy={accuracy:.3f} "
        f"threshold={args.threshold:.3f} "
        f"train={len(y_train)} "
        f"val={len(y_val)}"
    )

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
