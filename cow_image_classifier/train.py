"""
Training Script

Trains EfficientNet-B0 using two-phase transfer learning.

Phase 1:
    Freeze EfficientNet backbone.
    Train classifier only.

Phase 2:
    Unfreeze last EfficientNet blocks.
    Fine-tune the model.

Finally:
    Evaluate on the test dataset.
"""

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tqdm import tqdm

from cow_image_classifier import config
from cow_image_classifier.model import build_model

from dataset import build_dataloaders

from model import (
    build_model,
    unfreeze_last_blocks
)

# ==========================================================
# Run One Epoch
# ==========================================================

def run_epoch(
    model,
    dataloader,
    criterion,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    running_correct = 0

    total = 0

    all_predictions = []

    all_labels = []

    for images, labels in tqdm(
        dataloader,
        leave=False
    ):

        images = images.to(config.DEVICE)

        labels = labels.to(config.DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            predictions = outputs.argmax(dim=1)

            if training:

                loss.backward()

                optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

    epoch_loss = running_loss / total

    epoch_accuracy = running_correct / total

    return (

        epoch_loss,

        epoch_accuracy,

        all_predictions,

        all_labels

    )

# ==========================================================
# Test Model
# ==========================================================

def evaluate_test(
    model,
    test_loader,
    criterion,
    class_names
):

    print("\n" + "=" * 70)

    print("Testing Best Model")

    print("=" * 70)

    test_loss, test_accuracy, predictions, labels = run_epoch(

        model,

        test_loader,

        criterion,

        optimizer=None

    )

    print()

    print(f"Test Loss     : {test_loss:.4f}")

    print(f"Test Accuracy : {test_accuracy*100:.2f}%")

    print()

    print("=" * 70)

    print("Classification Report")

    print("=" * 70)

    print(

        classification_report(

            labels,

            predictions,

            target_names=class_names,

            digits=4

        )

    )

    print()

    print("=" * 70)

    print("Confusion Matrix")

    print("=" * 70)

    print(

        confusion_matrix(

            labels,

            predictions

        )

    )

# ==========================================================
# Train One Phase
# ==========================================================

def train_phase(

    model,

    train_loader,

    val_loader,

    criterion,

    optimizer,

    epochs,

    phase_name,

    best_state

):

    patience_counter = 0

    for epoch in range(epochs):

        print("\n" + "=" * 70)

        print(

            f"{phase_name} "

            f"Epoch {epoch+1}/{epochs}"

        )

        print("=" * 70)

        start = time.time()

        train_loss, train_acc, _, _ = run_epoch(

            model,

            train_loader,

            criterion,

            optimizer

        )

        val_loss, val_acc, _, _ = run_epoch(

            model,

            val_loader,

            criterion,

            optimizer=None

        )

        elapsed = time.time() - start

        print()

        print(

            f"Train Loss : {train_loss:.4f}"

        )

        print(

            f"Train Acc  : {train_acc*100:.2f}%"

        )

        print(

            f"Val Loss   : {val_loss:.4f}"

        )

        print(

            f"Val Acc    : {val_acc*100:.2f}%"

        )

        print(

            f"Time        : {elapsed:.1f} sec"

        )

        if val_acc > best_state["accuracy"]:

            best_state["accuracy"] = val_acc

            best_state["weights"] = copy.deepcopy(

                model.state_dict()

            )

            patience_counter = 0

            print("\nBest model updated.")

        else:

            patience_counter += 1

            if patience_counter >= config.PATIENCE:

                print(

                    f"\nEarly stopping "

                    f"after {config.PATIENCE} epochs."

                )

                break

    return best_state
# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("CowDoc AI - EfficientNet-B0 Training")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    train_loader, val_loader, test_loader, class_names = build_dataloaders()

    print("\nClasses")
    print(class_names)

    # ------------------------------------------------------
    # Build Model
    # ------------------------------------------------------

    model = build_model(
        num_classes=len(class_names)
    )

    criterion = nn.CrossEntropyLoss()

    best_state = {
        "accuracy": 0.0,
        "weights": None
    }

    # ======================================================
    # Phase 1
    # ======================================================

    print("\n" + "=" * 70)
    print("PHASE 1 : Training Classifier")
    print("=" * 70)

    optimizer = optim.Adam(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=config.PHASE1_LR,

        weight_decay=config.WEIGHT_DECAY

    )

    best_state = train_phase(

        model,

        train_loader,

        val_loader,

        criterion,

        optimizer,

        config.PHASE1_EPOCHS,

        "Phase 1",

        best_state

    )

    # ======================================================
    # Phase 2
    # ======================================================

    print("\n" + "=" * 70)
    print("PHASE 2 : Fine Tuning")
    print("=" * 70)

    model = unfreeze_last_blocks(

        model,

        config.UNFREEZE_LAST_N_BLOCKS

    )

    optimizer = optim.Adam(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=config.PHASE2_LR,

        weight_decay=config.WEIGHT_DECAY

    )

    best_state = train_phase(

        model,

        train_loader,

        val_loader,

        criterion,

        optimizer,

        config.PHASE2_EPOCHS,

        "Phase 2",

        best_state

    )

    # ======================================================
    # Save Best Model
    # ======================================================

    print("\n" + "=" * 70)
    print("Saving Best Model")
    print("=" * 70)

    model.load_state_dict(
        best_state["weights"]
    )

    checkpoint = {

        "model_state": model.state_dict(),

        "class_names": class_names,

        "accuracy": best_state["accuracy"]

    }

    torch.save(

        checkpoint,

        config.BEST_MODEL_PATH

    )

    print()

    print(
        f"Best Validation Accuracy : "
        f"{best_state['accuracy']*100:.2f}%"
    )

    print(
        f"Model Saved To : "
        f"{config.BEST_MODEL_PATH}"
    )

    # ======================================================
    # Test Evaluation
    # ======================================================

    evaluate_test(

        model,

        test_loader,

        criterion,

        class_names

    )

    print("\n" + "=" * 70)
    print("Training Completed Successfully")
    print("=" * 70)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()