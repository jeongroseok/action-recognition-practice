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


def _parse_timedelta(value: str) -> timedelta:
    """
    hh:mm:ss.s 형식의 문자열로부터 timedelta 파싱
    """
    arr = np.array(value.split(':')).astype('float')
    return timedelta(hours=arr[0], minutes=arr[1], seconds=arr[2])


def find_classes(root_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    `[class명] xxx/yyy/zzz.mp4 or .xml` 폴더 구조에서 class만 추출
    """
    classes = []
    for d in os.scandir(root_path):
        if not d.is_dir():
            continue
        c = d.name[1:3]
        if not c in classes:
            classes.append(c)
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(root_path: str, class_to_idx: Dict[str, int],
                 ext: Optional[Tuple[str, ...]]) -> List[Tuple[str, int]]:
    """
    `[class명] xxx/yyy/zzz.mp4 or .xml` 폴더 구조에서 `list(영상 경로, 클래스)` 변환
    """
    items = []
    directories = [d for d in os.scandir(root_path) if d.is_dir()]
    for d in directories:
        kls = class_to_idx[d.name[1:3]]
        for root, _, filenames in os.walk(d.path, followlinks=True):
            for fn in filenames:
                if fn.lower().endswith(ext):
                    items.append((os.path.join(root, fn), kls))
    return items


def create_clip(file_path: str, skip_sec: int = 5, scale: float = 0.5):
    """
    영상으로 부터 `skip_sec`만큼 건너띄며 클립 생성
    """
    frames = []
    event = et.parse(os.path.splitext(file_path)[0] + '.xml').find('event')
    starttime = _parse_timedelta(event.findtext('starttime')).total_seconds()
    duration = _parse_timedelta(event.findtext('duration')).total_seconds()
    sec = starttime
    cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        sec += skip_sec
        ret, frame = cap.read()
        if not ret or sec - starttime >= duration:
            cap.release()
            break
        frame = cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale)
        frames.append(frame)
    output = np.stack(frames)  # L, H, W, C
    print(f'{file_path} clip created')
    return np.einsum('ijkl->lijk', output)  # C, L, H, W


class CCTVDataset(Dataset):
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

        return sample, target


class CCTVDataModule(pl.LightningDataModule):
    def __init__(self, root_path: str, batch_size=32):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size

    def setup(self, stage):
        def to_tensor(x):
            x = torch.from_numpy(x)
            return x.float().div(255)

        transform = transforms.Compose([to_tensor])
        self.dataset = CCTVDataset(self.root_path, transform)

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
    ds = CCTVDataset(root_path)
    print(f"len(ds) = {len(ds)}")
    first = ds[0]
    print(f'first data = {first}')
    pass
