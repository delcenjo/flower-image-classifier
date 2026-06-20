from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGE_SIZE,
    MAX_PER_CLASS,
    SEED,
    TEST_SPLIT,
    VAL_SPLIT,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def capped_indices(targets, max_per_class, generator):
    if max_per_class is None:
        return list(range(len(targets)))
    buckets = defaultdict(list)
    for i in torch.randperm(len(targets), generator=generator).tolist():
        label = int(targets[i])
        if len(buckets[label]) < max_per_class:
            buckets[label].append(i)
    return [i for indices in buckets.values() for i in indices]


def build_dataloaders():
    generator = torch.Generator().manual_seed(SEED)
    train_base = datasets.ImageFolder(DATA_DIR, transform=train_transform())
    eval_base = datasets.ImageFolder(DATA_DIR, transform=eval_transform())

    indices = capped_indices(train_base.targets, MAX_PER_CLASS, generator)
    indices = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]

    n_test = int(len(indices) * TEST_SPLIT)
    n_val = int(len(indices) * VAL_SPLIT)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    loaders = (
        DataLoader(Subset(train_base, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(Subset(eval_base, val_idx), batch_size=BATCH_SIZE, num_workers=2),
        DataLoader(Subset(eval_base, test_idx), batch_size=BATCH_SIZE, num_workers=2),
    )
    return (*loaders, train_base.classes)
