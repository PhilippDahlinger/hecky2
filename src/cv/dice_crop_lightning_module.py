import torch
import torch.nn as nn
import wandb
from lightning import LightningModule
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models


class DiceCropModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.num_classes = config["num_classes"]

        # Load a pretrained backbone, e.g., MobileNetV3-Small
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features=in_features, out_features=self.num_classes)
        self.model = backbone

        # Multiclass loss (CrossEntropy expects logits and long class indices)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, average='macro')
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, average='macro')
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='macro')
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='macro')

        # Loss tracking
        self.training_step_outputs = []
        self.val_step_outputs = []

        self.train_misclassified_images = []  # Stores (image, pred, label)
        self.val_misclassified_images = []  # Stores (image, pred, label)

    def forward(self, x, output_logits=True):
        logits = self.model(x)
        return logits if output_logits else torch.softmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label'].squeeze(-1).long()  # Ensure shape [batch_size]
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.training_step_outputs.append(loss.item())

        # Collect misclassified images
        misclassified_mask = preds != y
        if misclassified_mask.any():
            misclassified_imgs = x[misclassified_mask]
            misclassified_preds = preds[misclassified_mask]
            misclassified_labels = y[misclassified_mask]
            for img, pred, label in zip(misclassified_imgs, misclassified_preds, misclassified_labels):
                if len(self.train_misclassified_images) < 30:
                    self.train_misclassified_images.append((img.cpu(), pred.item(), label.item()))

        return loss

    def on_train_epoch_end(self):
        import wandb

        train_loss = torch.tensor(self.training_step_outputs).mean()
        metrics = {
            'metrics/train_loss': train_loss,
            'metrics/train_acc': self.train_acc.compute(),
            'metrics/train_f1': self.train_f1.compute(),
            'trainer/epoch': self.current_epoch
        }
        vis = {}
        # Log misclassified images to wandb
        if self.train_misclassified_images:
            wandb_images = []
            for img, pred, label in self.train_misclassified_images:
                img = img.clone()
                wandb_images.append(
                    wandb.Image(img, caption=f"GT: {label}, Pred: {pred}")
                )
            metrics["vis/train_misclassified"] = wandb_images

        self.logger.experiment.log(metrics)

        # Reset
        self.training_step_outputs.clear()
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_misclassified_images.clear()

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label'].squeeze(-1).long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_step_outputs.append(loss.item())

        # Collect misclassified images
        misclassified_mask = preds != y
        if misclassified_mask.any():
            misclassified_imgs = x[misclassified_mask]
            misclassified_preds = preds[misclassified_mask]
            misclassified_labels = y[misclassified_mask]
            for img, pred, label in zip(misclassified_imgs, misclassified_preds, misclassified_labels):
                if len(self.val_misclassified_images) < 30:
                    self.val_misclassified_images.append((img.cpu(), pred.item(), label.item()))

        return loss

    def on_validation_epoch_end(self):
        val_loss = torch.tensor(self.val_step_outputs).mean()
        metrics = {
            'metrics/val_loss': val_loss,
            'metrics/val_acc': self.val_acc.compute(),
            'metrics/val_f1': self.val_f1.compute()
        }

        # Log misclassified images to wandb
        if self.val_misclassified_images:
            wandb_images = []
            for img, pred, label in self.val_misclassified_images:
                img = img.clone()
                wandb_images.append(
                    wandb.Image(img, caption=f"GT: {label}, Pred: {pred}")
                )
            metrics["vis/val_misclassified"] = wandb_images

        self.logger.experiment.log(metrics)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)

        # Reset
        self.val_step_outputs.clear()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_misclassified_images.clear()

    def configure_optimizers(self):
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
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param_groups[0]["params"].append(param)
            else:
                param_groups[1]["params"].append(param)

        optimizer = torch.optim.AdamW(param_groups)
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
