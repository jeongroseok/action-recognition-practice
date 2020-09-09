import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision import transforms
import os
from pytorch_lightning.metrics.functional import accuracy


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307, ), (0.3081, ))])

    train_loader = DataLoader(MNIST(download=True,
                                    root="./MNIST_data",
                                    transform=data_transform,
                                    train=True),
                              batch_size=train_batch_size)

    val_loader = DataLoader(MNIST(download=False,
                                  root="./MNIST_data",
                                  transform=data_transform,
                                  train=False),
                            batch_size=val_batch_size)
    return train_loader, val_loader


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1),
            nn.Softmax(1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        output = self.layer(x)
        output.squeeze_()
        return output

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3)
        return [opt], [sched]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return {'val_loss': loss, 'val_acc': acc}


from pytorch_lightning import Trainer, seed_everything
seed_everything(0)

# data
mnist_train = MNIST(os.getcwd(),
                    train=True,
                    download=True,
                    transform=transforms.ToTensor())
mnist_train = DataLoader(mnist_train, batch_size=32)
mnist_val = MNIST(os.getcwd(),
                  train=True,
                  download=True,
                  transform=transforms.ToTensor())
mnist_val = DataLoader(mnist_val, batch_size=11264)

# train_loader, val_loader = get_data_loaders(1024, 8192)
model = Net()
model.cuda()

model.layer = nn.DataParallel(model.layer)

# most basic trainer, uses good defaults
trainer = Trainer(progress_bar_refresh_rate=0.4, max_epochs=30, gpus=1)
trainer.fit(model, mnist_train, mnist_val)

import matplotlib.pyplot as plt
img, lbl = next(iter(mnist_val))
for i in img:
    preds = model(i.unsqueeze(0).cuda())
    plt.title(preds.argmax())
    plt.imshow(i.permute((1, 2, 0)))
    plt.show()

import torchvision.transforms.functional as F
F.to_tensor()