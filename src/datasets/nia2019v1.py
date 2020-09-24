import os
import xml.etree.ElementTree as et
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import cv2


def _onehot(labels: np.ndarray, num_classes: int):
    return np.eye(num_classes)[labels]


def _parse_timedelta(value: str) -> timedelta:
    """
    hh:mm:ss.s 형식의 문자열로부터 timedelta 파싱
    """
    arr = np.array(value.split(':')).astype('float')
    return timedelta(hours=arr[0], minutes=arr[1], seconds=arr[2])


class NIA2019V1Dataset(Dataset):
    """이상행동 CCTV 영상 AI데이터셋"""
    def __init__(self,
                 npz_filename: str,
                 transform: Optional[Callable] = None):

        self.transform = transform
        arr = np.load(npz_filename)

        self.clips = arr["clips"]
        self.targets = arr["targets"]

        self.__classes, self.__cls_to_idx = self.__find_classes(self.targets)
        self.__idx_to_cls = dict(
            (value, key) for key, value in self.__cls_to_idx.items())

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        clip = self.clips[idx]

        if self.transform is not None:
            clip = self.transform(clip)

        return clip, torch.tensor(
            self.__cls_to_idx[self.targets[idx]],
            dtype=torch.long)  # _onehot(target, self.num_classes)

    def idx_to_cls(self, idx: int) -> str:
        if type(idx) == torch.Tensor:
            idx = idx.item()
        return self.__idx_to_cls[idx]

    def cls_to_idx(self, cls: str) -> int:
        return self.cls_to_idx()[cls]

    @staticmethod
    def __find_classes(
            targets: np.ndarray) -> Tuple[List[str], Dict[str, int]]:
        classes = np.unique(targets).tolist()
        cls_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, cls_to_idx

    @property
    def classes(self):
        return self.__classes


class NIA2019V1DataModule(pl.LightningDataModule):
    def __init__(self, npz_filename: str, batch_size: int = 1):
        super().__init__()
        self.npz_filename = npz_filename
        self.batch_size = batch_size

    def setup(self, stage):
        def to_tensor(x):
            x = torch.from_numpy(x)
            return x.float().div(255)

        transform = transforms.Compose([to_tensor])
        self.dataset = NIA2019V1Dataset(self.npz_filename, transform)

    def train_dataloader(self) -> Dataset:
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self) -> Dataset:
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

    def test_dataloader(self) -> Dataset:
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
