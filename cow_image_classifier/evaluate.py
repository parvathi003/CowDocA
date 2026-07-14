"""
Evaluate Trained Model

Loads the best saved EfficientNet-B0 model
and evaluates it on the test dataset.
"""

import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import config

from dataset import build_dataloaders
from model import build_model

# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Loading Test Dataset")
    print("=" * 60)

    _, _, test_loader, class_names = build_dataloaders()

    print("\nLoading Best Model...\n")

    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE
    )

    model = build_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    predictions = []

    labels_list = []

    running_loss = 0.0

    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(config.DEVICE)

            labels = labels.to(config.DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)

            predictions.extend(
                predicted.cpu().numpy()
            )

            labels_list.extend(
                labels.cpu().numpy()
            )

            total += images.size(0)

    test_loss = running_loss / total

    accuracy = accuracy_score(
        labels_list,
        predictions
    )

    print("=" * 60)
    print("Test Results")
    print("=" * 60)

    print(f"\nTest Loss     : {test_loss:.4f}")

    print(f"Test Accuracy : {accuracy*100:.2f}%")

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)

    print(

        classification_report(

            labels_list,

            predictions,

            target_names=class_names,

            digits=4

        )

    )

    print("=" * 60)
    print("Confusion Matrix")
    print("=" * 60)

    print(

        confusion_matrix(

            labels_list,

            predictions

        )

    )


# ==========================================================

if __name__ == "__main__":

    main()