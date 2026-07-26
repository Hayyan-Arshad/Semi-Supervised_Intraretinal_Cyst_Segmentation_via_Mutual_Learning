import itertools
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class OCTH5Dataset(Dataset):
    """SSL4MIS-style HDF5 dataset for OCT segmentation."""

    def __init__(self, root, split="train", transform=None, max_samples=None, intensity_norm="zscore", eps=1e-6):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.intensity_norm = intensity_norm
        self.eps = eps
        list_name = "train_slices.list" if split == "train" else "val.list"
        with (self.root / list_name).open("r") as handle:
            self.sample_list = [line.strip() for line in handle if line.strip()]
        if max_samples is not None and split == "train":
            self.sample_list = self.sample_list[:max_samples]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            path = self.root / "data" / "slices" / f"{case}.h5"
        else:
            path = self.root / "data" / f"{case}.h5"
        with h5py.File(path, "r") as handle:
            image = normalize_image(handle["image"][:], self.intensity_norm, self.eps)
            label = handle["label"][:]
        sample = {"image": image, "label": label, "idx": idx}
        if self.transform is not None:
            sample = self.transform(sample)
            sample["idx"] = idx
        return sample


def normalize_image(image, mode="zscore", eps=1e-6):
    image = image.astype(np.float32)
    if mode in (None, "none"):
        return image
    if mode == "minmax":
        image_min = np.min(image)
        image_max = np.max(image)
        return (image - image_min) / max(float(image_max - image_min), eps)
    if mode == "zscore":
        return (image - float(np.mean(image))) / max(float(np.std(image)), eps)
    if mode == "clip_zscore":
        normalized = (image - float(np.mean(image))) / max(float(np.std(image)), eps)
        return np.clip(normalized, -5.0, 5.0) / 5.0
    raise ValueError(f"Unsupported intensity_norm '{mode}'. Available: none, minmax, zscore, clip_zscore")


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() < 0.5:
            image, label = random_rot_flip(image, label)
        else:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.uint8)),
        }


class TwoStreamBatchSampler(Sampler):
    """Iterate labeled and unlabeled indices in each batch."""

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = np.random.permutation(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for primary_batch, secondary_batch in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
