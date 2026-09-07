from pathlib import Path

import numpy as np
from h5py import File


def episode_split(
    data_path: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    with File(data_path, "r") as f:
        success = f["episode_meta"]["success"][:].astype(bool)

    rng = np.random.default_rng(seed)
    train, val = [], []

    # split failures and successes separately so both land in each half
    for group in (np.flatnonzero(~success), np.flatnonzero(success)):
        rng.shuffle(group)
        n = round(len(group) * val_fraction)
        val.append(group[:n])
        train.append(group[n:])

    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))
