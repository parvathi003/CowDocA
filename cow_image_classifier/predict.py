"""
Prediction Script

Loads the trained EfficientNet-B0 model and predicts
the disease from a cow image.
"""

import torch
from PIL import Image
from torchvision import transforms

import config
from model import build_model

# ==========================================================
# ImageNet Normalization
# ==========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================================
# Load Model
# ==========================================================

_model = None
_class_names = None


def load_model():

    global _model
    global _class_names

    if _model is not None:
        return _model, _class_names

    checkpoint = torch.load(

        config.BEST_MODEL_PATH,

        map_location=config.DEVICE

    )

    class_names = checkpoint["class_names"]

    model = build_model(

        num_classes=len(class_names)

    )

    model.load_state_dict(

        checkpoint["model_state"]

    )

    model.eval()

    _model = model

    _class_names = class_names

    return model, class_names


# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(config.IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=IMAGENET_MEAN,

        std=IMAGENET_STD

    )

])


# ==========================================================
# Prediction
# ==========================================================

def predict_disease(image_path, top_k=3):

    model, class_names = load_model()

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(config.DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(

            outputs,

            dim=1

        )

    top_probabilities, top_indices = torch.topk(

        probabilities,

        k=min(top_k, len(class_names))

    )

    results = []

    for probability, index in zip(

        top_probabilities[0],

        top_indices[0]

    ):

        results.append({

            "disease": class_names[index],

            "confidence": round(

                probability.item() * 100,

                2

            )

        })

    return results


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print()

        print("Usage")

        print()

        print("python predict.py image.jpg")

    else:

        predictions = predict_disease(

            sys.argv[1]

        )

        print()

        print("=" * 60)

        print("Prediction")

        print("=" * 60)

        print()

        for prediction in predictions:

            print(

                f"{prediction['disease']:<15}"

                f"{prediction['confidence']:.2f}%"

            )