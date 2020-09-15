import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import Trainer


class R2Plus1DEncoder(pl.LightningModule):
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


class DecoderLSTM(pl.LightningModule):
    """
    https://github.com/HHTseng/video-classification/blob/master/ResNetCRNN/functions.py
    """
    def __init__(self,
                 CNN_embed_dim=300,
                 h_RNN_layers=3,
                 h_RNN=256,
                 h_FC_dim=128,
                 drop_p=0.3,
                 num_classes=50):
        super().__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=
            True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x