import pickle
from pathlib import Path
import numpy as np
from einops import rearrange

class CIFARSource:
    def __init__(self, folder_name, file_prefix):
        current_dir = Path(folder_name)
        files_to_read = sorted([f for f in current_dir.iterdir() if
                                f.name.startswith(file_prefix)])

        self.data = []
        self.labels = []

        for f in files_to_read:
            with open(f, 'rb') as fo:
                d = pickle.load(fo, encoding="bytes")
                self.data.append(d[b"data"])
                self.labels.append(d[b"labels"])

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        self.data = rearrange(self.data,
                              "batch (channel height width) -> batch height width channel",
                              height=32,
                              width=32,
                              channel=3)

        self.data = np.divide(self.data, 255, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"img": self.data[idx],
                "label": self.labels[idx]}
