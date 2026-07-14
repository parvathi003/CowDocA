"""
Dataset Loader

Loads the training, validation and testing datasets
for the CowDoc AI image classification model.
"""

import json
import os
from collections import Counter

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import config

# ==========================================================
# ImageNet Normalization
# ==========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================================
# Image Transforms
# ==========================================================


def get_transforms():

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            config.IMG_SIZE,
            scale=(0.8, 1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    ])

    test_transform = transforms.Compose([

        transforms.Resize(
            int(config.IMG_SIZE * 1.14)
        ),

        transforms.CenterCrop(
            config.IMG_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    ])

    return train_transform, test_transform


# ==========================================================
# Build DataLoaders
# ==========================================================


def build_dataloaders():

    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        config.TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        config.VAL_DIR,
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        config.TEST_DIR,
        transform=test_transform
    )

    # ------------------------------------------------------
    # Verify class names
    # ------------------------------------------------------

    assert (
        train_dataset.classes ==
        val_dataset.classes ==
        test_dataset.classes
    ), "Class folders do not match."

    class_names = train_dataset.classes

    # ------------------------------------------------------
    # Weighted Sampling
    # ------------------------------------------------------

    targets = [label for _, label in train_dataset.samples]

    class_counts = Counter(targets)

    class_weights = {

        cls: 1.0 / (count ** 0.5)

        for cls, count in class_counts.items()

    }

    sample_weights = [

        class_weights[label]

        for label in targets

    ]

    sampler = WeightedRandomSampler(

        sample_weights,

        num_samples=len(sample_weights),

        replacement=True

    )

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.BATCH_SIZE,

        sampler=sampler,

        num_workers=config.NUM_WORKERS,

        pin_memory=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=True

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=True

    )

    # ------------------------------------------------------
    # Save Class Names
    # ------------------------------------------------------

    os.makedirs(

        config.CHECKPOINT_DIR,

        exist_ok=True

    )

    with open(

        config.CLASS_NAMES_PATH,

        "w"

    ) as f:

        json.dump(

            class_names,

            f,

            indent=4

        )

    print("\n" + "=" * 60)

    print("Dataset Loaded Successfully")

    print("=" * 60)

    print()

    print("Classes")

    print(class_names)

    print()

    print("Number of Classes")

    print(len(class_names))

    print()

    print("Training Images")

    print(len(train_dataset))

    print()

    print("Validation Images")

    print(len(val_dataset))

    print()

    print("Testing Images")

    print(len(test_dataset))

    print()

    counts_readable = {

        class_names[k]: v

        for k, v in class_counts.items()

    }

    print("Training Images Per Class")

    print(counts_readable)

    print()

    return (

        train_loader,

        val_loader,

        test_loader,

        class_names

    )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    build_dataloaders()