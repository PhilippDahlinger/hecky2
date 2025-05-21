import os

import pandas as pd
import torch
import traceback
import sys

import wandb
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from cv.custom_wandb_logger import CustomWandBLogger
from cv.data.dice_crop_dataset import DiceCropDataset
from cv.dice_crop_lightning_module import DiceCropModel


def train(config) -> None:
    train_manifest = pd.read_csv(os.path.join("data", "dice_crop_dataset", "train.csv"))
    val_manifest = pd.read_csv(os.path.join("data", "dice_crop_dataset", "test.csv"))
    train_ds = DiceCropDataset(train_manifest, data_augmentation=True)
    eval_ds = DiceCropDataset(val_manifest, data_augmentation=False)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=64, shuffle=False)

    lit_model = DiceCropModel(config)

    # Initialize WandB logger
    wandb_logger = CustomWandBLogger(
        project="Hecky2_dice_crop",  # Name of your WandB project
        name=config["exp_name"],  # Name of the current run
        group="group",  # Group name for the run
        config=config,  # Configuration object
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor
        dirpath=os.path.join("runs", "dice_crop", config["exp_name"], "checkpoints"),  # Directory to save checkpoints
        filename='best-checkpoint-{epoch:02d}-{val_loss:.8f}',  # Checkpoint filename format
        save_top_k=3,  # Save only the best `k` models (1 = best model only)
        mode='min',  # Mode for monitoring ('min' for lower is better, 'max' for higher is better)
        save_last=True  # Optionally save the most recent model
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = []
    callbacks.append(checkpoint_callback)
    if wandb_logger:
        callbacks.append(learning_rate_monitor)
    if len(callbacks) == 0:
        callbacks = None

    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=wandb_logger,  # Use the wandb logger
        max_epochs=100,  # Max number of epochs for training
        accelerator="cpu",  # what type of accelerator to use
        devices=1,  # how many devices to use (if accelerator is not None)
        callbacks=callbacks,  # Checkpointing callback
        default_root_dir=os.path.join("runs", "dice_crop", config["exp_name"]),  # Where to save logs and checkpoints
        enable_checkpointing=True,  # Enable checkpointing
    )

    # Now, start the training
    trainer.fit(lit_model, train_dataloaders=train_dl, val_dataloaders=eval_dl)
    # Call wandb finish.
    wandb.finish()


if __name__ == '__main__':
    config = yaml.safe_load(open("config/dice_crop.yaml", "r"))
    train(config)
