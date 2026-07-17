"""
Model Definition

Builds the EfficientNet-B0 model used for cattle disease classification.
"""

import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from cow_image_classifier import config


# ==========================================================
# Build Model
# ==========================================================

def build_model(num_classes):
    """
    Creates an EfficientNet-B0 model using transfer learning.

    Phase 1:
        Freeze the EfficientNet backbone.
        Train only the classifier.

    Phase 2:
        Unfreeze the last few feature blocks for fine-tuning.
    """

    # ------------------------------------------------------
    # Load pretrained EfficientNet-B0
    # ------------------------------------------------------

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1

    model = efficientnet_b0(weights=weights)

    # ------------------------------------------------------
    # Freeze Feature Extractor
    # ------------------------------------------------------

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # ------------------------------------------------------
    # Replace Classifier
    # ------------------------------------------------------

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(
            in_features,
            num_classes
        )

    )

    return model.to(config.DEVICE)


# ==========================================================
# Fine Tuning
# ==========================================================

def unfreeze_last_blocks(model, n_blocks):
    """
    Unfreezes the last n EfficientNet feature blocks.

    Earlier layers remain frozen because they already
    know general image features such as edges, textures,
    and colours.

    The final blocks learn cattle-specific disease
    features.
    """

    total_blocks = len(model.features)

    start_block = max(
        0,
        total_blocks - n_blocks
    )

    for block_index, block in enumerate(model.features):

        if block_index >= start_block:

            for parameter in block.parameters():

                parameter.requires_grad = True

    return model


# ==========================================================
# Count Parameters
# ==========================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    model = build_model(
        num_classes=6
    )

    total, trainable = count_parameters(model)

    print("=" * 60)
    print("EfficientNet-B0")
    print("=" * 60)

    print()

    print(f"Device                 : {config.DEVICE}")

    print(f"Total Parameters       : {total:,}")

    print(f"Trainable Parameters   : {trainable:,}")

    print()

    model = unfreeze_last_blocks(
        model,
        config.UNFREEZE_LAST_N_BLOCKS
    )

    total, trainable = count_parameters(model)

    print("After Fine-Tuning")

    print(f"Trainable Parameters   : {trainable:,}")

    print()

    print("Model created successfully.")