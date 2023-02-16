import os
from utils import BinaryModule
import torch
import pytorch_lightning as pl
import wandb
from dotenv import load_dotenv


config = {
    "classifier_eval_mode": True,
    "criterion": torch.nn.BCELoss(),
    "optimizer": torch.optim.Adam,
    "optimizer_params": {
        "lr": 0.005,
    },
    "dataset_params": {
        "data_path": "data/",
        "split_size": 6000,
    },
    "dataloader_params": {
        "batch_size": 12,
    },
}

if __name__ == "__main__":
    load_dotenv(".env")
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    module = BinaryModule(config)
    logger = pl.loggers.WandbLogger(project="binary_quantization", name="test")
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=500,
        max_epochs=3,
    )
    trainer.fit(module)
    trainer.test()
