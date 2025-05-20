import torch
import torch.nn as nn
from lightning import LightningModule
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models


class DiceCropClassifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.num_classes = config.num_classes

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

    def forward(self, x, output_logits=True):
        logits = self.model(x)
