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
    def __init__(self, root_path: str, transform: Optional[Callable] = None):

        self.root_path = root_path
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(self.root_path)

        self.samples = make_dataset(self.root_path, self.class_to_idx, ('mp4'))
        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = create_clip(path)  # L, C, H, W
        sample = np.array(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, torch.tensor(
            target, dtype=torch.long)  # _onehot(target, self.num_classes)


class NIA2019V1DataModule(pl.LightningDataModule):
    def __init__(self, root_path: str, batch_size=1):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size

    def setup(self, stage):
        def to_tensor(x):
            x = torch.from_numpy(x)
            return x.float().div(255)

        transform = transforms.Compose([to_tensor])
        self.dataset = NIA2019V1Dataset(self.root_path, transform)

    # return the dataloader for each split
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    global root_path, ds
    root_path = "E:\subset"
    ds = NIA2019V1Dataset(root_path)
    print(f"len(ds) = {len(ds)}")
    first = ds[0]
    print(f'first data = {first}')
    pass
