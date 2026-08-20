from pathlib import Path
import poke.env
import poke.actions

from torch.utils.data import Dataset

class PokeDataset(Dataset):
    def __init__(self, data_path: str | Path) -> None:
        ...