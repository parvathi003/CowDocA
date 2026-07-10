import os
import random
import shutil
from pathlib import Path

# =====================================================
# Configuration
# =====================================================

SOURCE_DIR = "Image_Disease"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(42)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)

# =====================================================
# Create Output Folder
# =====================================================

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# =====================================================
# Process Each Disease Folder
# =====================================================

for disease in os.listdir(SOURCE_DIR):

    disease_path = os.path.join(SOURCE_DIR, disease)

    if not os.path.isdir(disease_path):
        continue

    images = [

        f for f in os.listdir(disease_path)

        if f.lower().endswith(IMAGE_EXTENSIONS)

    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # ------------------------------------------

    for split_name, split_images in [

        ("train", train_images),
        ("val", val_images),
        ("test", test_images)

    ]:

        split_folder = os.path.join(
            OUTPUT_DIR,
            split_name,
            disease
        )

        Path(split_folder).mkdir(
            parents=True,
            exist_ok=True
        )

        for image in split_images:

            src = os.path.join(
                disease_path,
                image
            )

            dst = os.path.join(
                split_folder,
                image
            )

            shutil.copy2(src, dst)

    print(
        f"{disease:<12} "
        f"Train:{len(train_images):4} "
        f"Val:{len(val_images):4} "
        f"Test:{len(test_images):4}"
    )

print("\nDataset split completed successfully!")