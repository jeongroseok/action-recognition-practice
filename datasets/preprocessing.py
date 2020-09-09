from .dataset import *
import numpy as np
from time import time


def preprocess_raw_data(root_path: str) -> np.ndarray:
    classes, class_to_idx = find_classes(root_path)
    samples = make_dataset(root_path, class_to_idx, ('mp4'))

    clips = []
    targets = []
    num_samples = len(samples)
    cnt = 0
    starttime = time()
    for file, target in samples:
        targets.append(target)
        clips.append(create_clip(file))
        cnt += 1
        print(f"{cnt}/{num_samples} complete, {(cnt / num_samples) * 100}%")
        elapsed = time() - starttime
        avg = elapsed / cnt
        print(f"avg {avg}, estimation {avg * num_samples}")

    clips = [create_clip(file_path) for file_path, _ in samples]
    targets = [target for _, target in samples]
    return np.array((clips, targets))


def save_processed_data(file: str, a: np.ndarray):
    np.savez_compressed(file, a)


def load_processed_data(file: str) -> np.ndarray:
    return np.load(file)