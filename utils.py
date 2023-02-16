from pathlib import Path
import numpy as np
import pandas as pd
import torch


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: Path, train_mode: bool = True, split_size: int = None
    ):
        self.data_path = data_path
        self.split_size = split_size

        self.df = pd.read_csv(
            self.data_path + "binary_train.tsv",
            sep="\t",
            names=["filenames", "labels"],
            dtype={"filenames": str},
        )

        self._train_folder = "data/binary_train/"
        self.df.filenames = self._train_folder + self.df.filenames + ".npy"
        self.df = self.df[self.split_size :]

        self._train_mode = train_mode

        if self._train_mode:
            self.df = self.df[:split_size]

    def __getitem__(self, idx):
        image_path, label = self.df.loc[idx]
        image = np.load(image_path)
        image = torch.from_numpy(image)
        image = torch.permute(image, dims=(1, 0))
        image = image.to(torch.float32)

        return image, label

    def __len__(self):
        return len(self.df)
