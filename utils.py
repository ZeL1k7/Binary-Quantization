from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
)
import pytorch_lightning as pl
from model import BinaryModel


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
        self._train_mode = train_mode

        if self._train_mode:
            self.df = self.df.sample(self.split_size)
        else:
            self.df = self.df.sample(len(self.df) - self.split_size)

    def __getitem__(self, idx):
        image_path, label = self.df.iloc[idx]

        image = np.load(image_path)
        image = torch.from_numpy(image)
        image = torch.permute(image, dims=(1, 0))
        image = image.to(torch.float32)

        label = torch.tensor(label)
        label = label.unsqueeze(-1)
        label = label.to(torch.float32)

        return image, label

    def __len__(self):
        return len(self.df)


def collate_batch(batch):
    max_len = max([b[0].size(1) for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    padder = lambda x: torch.nn.functional.pad(
        x, (0, max_len - x.size(1)), "constant", 0
    )
    padded_images = tuple(padder(b[0]) for b in batch)
    padded_images = torch.stack(padded_images, 0)
    padded_image = torch.permute(padded_images, (1, 0, 2))
    return padded_images, labels


class BinaryModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.model = BinaryModel(in_channels=64, hid_channels=128, out_channels=1)
        self.criterion = self.get_criterion()
        self.rocauc_calculator = BinaryAUROC()
        self.accuracy_calculator = BinaryAccuracy()
        self.recall_calculator = BinaryRecall()
        self.precision_calculator = BinaryPrecision()

    def train_dataset(self):
        params = self._config["dataset_params"]
        train_dataset = BinaryDataset(train_mode=True, **params)
        return train_dataset

    def train_dataloader(self):
        dataset = self.train_dataset()
        params = self._config["dataloader_params"]
        return torch.utils.data.DataLoader(
            dataset, shuffle=True, collate_fn=collate_batch, **params
        )

    def test_dataset(self):
        params = self._config["dataset_params"]
        test_dataset = BinaryDataset(train_mode=False, **params)
        return test_dataset

    def test_dataloader(self):
        dataset = self.test_dataset()
        params = self._config["dataloader_params"]
        return torch.utils.data.DataLoader(
            dataset, shuffle=False, collate_fn=collate_batch, **params
        )

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        params = self._config["optimizer_params"]
        optimizer = self._config["optimizer"](self.model.parameters(), **params)
        return optimizer

    def get_criterion(self):
        criterion = self._config["criterion"]
        return criterion

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.model(inputs)
        loss = self.criterion(logits, labels.unsqueeze(-1))
        self.log("loss", loss)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.model(inputs)
        return {"logits": logits.cpu(), "labels": labels.cpu()}

    def test_epoch_end(self, outputs) -> None:
        logits = torch.stack([b["logits"] for b in outputs], 0)
        labels = torch.stack([b["labels"] for b in outputs], 1)
        labels = labels.unsqueeze(1)
        labels = torch.permute(labels, (0, 2, 1))

        rocauc_metrics = self.rocauc_calculator(logits, labels)
        accuracy_metrics = self.accuracy_calculator(logits, labels)
        recall_metrics = self.recall_calculator(logits, labels)
        precision_metrics = self.precision_calculator(logits, labels)

        self.log("rocauc_metrics", rocauc_metrics)
        self.log("accuracy_metrics", accuracy_metrics)
        self.log("recall_metrics", recall_metrics)
        self.log("precision_metrics", precision_metrics)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs) -> None:
        logits = torch.stack([b["logits"] for b in outputs], 0)
        labels = torch.stack([b["labels"] for b in outputs], 0)
        labels = labels.unsqueeze(1)
        labels = torch.permute(labels, (0, 2, 1))

        rocauc_metrics = self.rocauc_calculator(logits, labels)
        accuracy_metrics = self.accuracy_calculator(logits, labels)
        recall_metrics = self.recall_calculator(logits, labels)
        precision_metrics = self.precision_calculator(logits, labels)

        self.log("rocauc_metrics", rocauc_metrics)
        self.log("accuracy_metrics", accuracy_metrics)
        self.log("recall_metrics", recall_metrics)
        self.log("precision_metrics", precision_metrics)
