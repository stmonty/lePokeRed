import hashlib
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

H, W = 144, 160


class EpisodeWriter:
    def __init__(self, path, rom, act_frames, chunk=64):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(path, "w")
        self.h5.attrs["rom_sha1"] = hashlib.sha1(Path(rom).read_bytes()).hexdigest()
        self.h5.attrs["pyboy_version"] = version("pyboy")
        self.h5.attrs["act_frames"] = act_frames

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

        self.n = 0
        self._buf = []

    def add_step(self, frame, action, party_count):
        self._buf.append((frame, action, party_count))

    def end_episode(self):
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

        self.n += t
        self._buf = []

    def close(self):
        self.end_episode()
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
