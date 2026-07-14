"""
Configuration File

Stores all project settings used by the image classification module.
"""

import os
import torch

# ==========================================================
# Base Directory
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================
# Dataset Paths
# ==========================================================

DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ==========================================================
# Checkpoint Paths
# ==========================================================

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pt"
)

CLASS_NAMES_PATH = os.path.join(
    CHECKPOINT_DIR,
    "class_names.json"
)

# ==========================================================
# Image Settings
# ==========================================================

IMG_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 2

# ==========================================================
# Training Settings
# ==========================================================

# Phase 1
PHASE1_EPOCHS = 8
PHASE1_LR = 1e-3

# Phase 2
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5

# Regularization
WEIGHT_DECAY = 1e-4

# Fine-tuning
UNFREEZE_LAST_N_BLOCKS = 2

# Early Stopping
PATIENCE = 5

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# Create Checkpoint Folder Automatically
# ==========================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)