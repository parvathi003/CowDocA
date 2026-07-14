"""
Split Dataset

Converts

Image_Disease/
    FMD/
    FootRot/
    Healthy/
    LSD/
    Mastitis/
    Ringworm/

into

dataset/
    train/
    val/
    test/

Original images are NOT modified.
"""

import os
import random
import shutil

# ==========================================================
# Configuration
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "..",
    "Image_Disease"
)

DEST_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

SEED = 42

random.seed(SEED)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)

# ==========================================================
# Main
# ==========================================================


def main():

    if not os.path.exists(SOURCE_DIR):
        print(f"\nSource folder not found:\n{SOURCE_DIR}")
        return

    # Delete old dataset if it exists
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    class_names = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])

    print("=" * 60)
    print("Found Classes")
    print("=" * 60)
    print(class_names)
    print()

    for cls in class_names:

        src_folder = os.path.join(SOURCE_DIR, cls)

        images = []

        # Read complete paths directly
        for entry in os.scandir(src_folder):

            if not entry.is_file():
                continue

            if entry.name.lower().endswith(IMAGE_EXTENSIONS):
                images.append(entry.path)

        random.shuffle(images)

        total = len(images)

        if total == 0:
            print(f"{cls}: No images found.")
            continue

        train_end = int(total * TRAIN_SPLIT)
        val_end = train_end + int(total * VAL_SPLIT)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        splits = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]

        copied = {
            "train": 0,
            "val": 0,
            "test": 0
        }

        for split_name, files in splits:

            split_folder = os.path.join(
                DEST_DIR,
                split_name,
                cls
            )

            os.makedirs(
                split_folder,
                exist_ok=True
            )

            for src in files:

                filename = os.path.basename(src)

                dst = os.path.join(
                    split_folder,
                    filename
                )

                try:
                    shutil.copy2(src, dst)
                    copied[split_name] += 1

                except Exception as e:
                    print(f"Skipped : {filename}")
                    print(e)

        print(
            f"{cls:<12}"
            f"Train: {copied['train']:<5}"
            f"Val: {copied['val']:<5}"
            f"Test: {copied['test']:<5}"
            f"Total: {total}"
        )

    print("\n" + "=" * 60)
    print("Dataset split completed successfully!")
    print("=" * 60)


# ==========================================================

if __name__ == "__main__":
    main()