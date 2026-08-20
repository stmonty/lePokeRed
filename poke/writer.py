import hashlib
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

from poke.actions import BUTTON_ORDER

H, W = 144, 160

EPISODE_DTYPE = np.dtype([
    ("kind", "S8"),
    ("name", "S24"),
    ("success", "u1"),
    ("species", "i2"),
])


def _sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


class EpisodeWriter:
    def __init__(self, path, rom, state, act_frames, seed, chunk=64):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(path, "w")
        self.h5.attrs["rom_sha1"] = _sha1(rom)
        self.h5.attrs["state_sha1"] = _sha1(state)
        self.h5.attrs["pyboy_version"] = version("pyboy")
        self.h5.attrs["act_frames"] = act_frames
        self.h5.attrs["seed"] = seed
        self.h5.attrs["button_order"] = BUTTON_ORDER

        gz = {"compression": "gzip", "compression_opts": 9}
        self.frames = self.h5.create_dataset(
            "frames", shape=(0, H, W, 1), maxshape=(None, H, W, 1),
            dtype=np.uint8, chunks=(chunk, H, W, 1), **gz,
        )
        self.actions = self.h5.create_dataset(
            "actions", shape=(0, 6), maxshape=(None, 6), dtype=np.uint8,
        )
        self.party = self.h5.create_dataset(
            "party_counts", shape=(0,), maxshape=(None,), dtype=np.uint8,
        )
        self.episodes = self.h5.create_dataset(
            "episodes", shape=(0, 2), maxshape=(None, 2), dtype=np.int64,
        )
        self.meta = self.h5.create_dataset(
            "episode_meta", shape=(0,), maxshape=(None,), dtype=EPISODE_DTYPE,
        )

        self.n = 0
        self._buf = []

    def add_step(self, frame, action, party_count):
        self._buf.append((frame, action, party_count))

    def end_episode(self, kind="", name="", success=False, species=-1):
        if not self._buf:
            return
        t = len(self._buf)
        frames = np.stack([b[0] for b in self._buf]).reshape(t, H, W, 1)
        actions = np.stack([b[1] for b in self._buf]).astype(np.uint8)
        party = np.array([b[2] for b in self._buf], dtype=np.uint8)

        for ds, data in ((self.frames, frames), (self.actions, actions), (self.party, party)):
            ds.resize(self.n + t, axis=0)
            ds[self.n:] = data

        e = self.episodes.shape[0]
        self.episodes.resize(e + 1, axis=0)
        self.episodes[e] = (self.n, t)
        self.meta.resize(e + 1, axis=0)
        self.meta[e] = (kind.encode(), name.encode(), int(success), species)

        self.n += t
        self._buf = []

    def close(self):
        self.end_episode()
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
