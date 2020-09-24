import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast


class PracticeModel(pl.LightningModule):
    """
    (batch, channel, time, height, width)
    """
    def __init__(self, num_classes: int = 128):
        super().__init__()
        self.model = models.video.r2plus1d_18(True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3)
        return [opt], [sched]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print(f"y_hat: {y_hat}, y: {y}")
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(torch.eye(5)[y_hat], y)
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print(f"y_hat: {y_hat}, y: {y}")
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}
