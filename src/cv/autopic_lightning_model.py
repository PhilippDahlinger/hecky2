import torch
import torch.nn as nn
from lightning import LightningModule
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models


class AutopicModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        # Load for example the pretrained MobileNetV3-Small backbone
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        in_features = backbone.classifier[-1].in_features
        # Replace classifier with custom head
        backbone.classifier[-1] = nn.Linear(in_features=in_features, out_features=1)
        # these are logits, to get the 0,1 output value, use sigmoid

        self.model = backbone
        # Define loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

        # temp lists to save loss results
        self.training_step_outputs = []
        self.val_step_outputs = []

    def forward(self, x, output_logits=True):
        # model outputs logits, so we need to apply sigmoid to get the probabilities
        if output_logits:
            return self.model(x)
        else:
            return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch["label"]
        logits = self(x, output_logits=True)
        train_loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.train_f1(logits, y)
        # save in the temp dict
        self.training_step_outputs.append(train_loss.item())
        return train_loss

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from the `training_step` for each batch
        # Calculate the mean loss for the epoch
        loss_epoch_average = torch.tensor(self.training_step_outputs).mean()
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        # Log the results to wandb
        self.log_dict({
            'train_loss': loss_epoch_average,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'epoch': self.current_epoch
        }, on_epoch=True, prog_bar=True)
        # reset the metrics
        self.training_step_outputs.clear()  # free memory
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch["label"]
        logits = self(x, output_logits=True)
        val_loss = self.criterion(logits, y)

        # Update metrics
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        # save in the temp dict
        self.val_step_outputs.append(val_loss.item())
        return val_loss

    def on_validation_epoch_end(self):
        val_loss = torch.tensor(self.val_step_outputs).mean()
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        self.log_dict({
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
        }, on_epoch=True, prog_bar=True)
        # Reset for next epoch
        self.val_step_outputs.clear()
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        # Set up parameter groups
        param_groups = [
            {
                "params": [],
                "lr": self.hparams.lr_classifier,
                "weight_decay": self.hparams.weight_decay_classifier,
            },
            {
                "params": [],
                "lr": self.hparams.lr_backbone,
                "weight_decay": self.hparams.weight_decay_backbone,
            },
        ]

        # Assign parameters to the appropriate group
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param_groups[0]["params"].append(param)
            else:
                param_groups[1]["params"].append(param)

        # Create the optimizer
        optimizer = torch.optim.AdamW(param_groups)

        # Set up cosine annealing scheduler
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_scheduler["T_max"],
                eta_min=self.hparams.lr_scheduler["eta_min"]
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "cosine_lr"
        }
        return [optimizer], [scheduler]
