"""
Prediction Script

Loads the trained EfficientNet-B0 model and predicts
the disease from a cow image.

Supports:
- Image file path
- Streamlit UploadedFile
- PIL Image
"""

import torch
from PIL import Image
from torchvision import transforms

from cow_image_classifier import config
from cow_image_classifier.model import build_model

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
# Load Image
# ==========================================================

def load_image(image_input):
    """
    Accepts

    - image path
    - Streamlit UploadedFile
    - PIL Image
    """

    if isinstance(image_input, Image.Image):

        return image_input.convert("RGB")

    return Image.open(image_input).convert("RGB")


# ==========================================================
# Prediction
# ==========================================================

def predict_disease(
    image_input,
    top_k=3
):

    model, class_names = load_model()

    image = load_image(image_input)

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

    predictions = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        predictions.append({

            "disease": class_names[index],

            "confidence": round(
                probability.item() * 100,
                2
            )

        })

    return {

        "top_prediction": predictions[0],

        "predictions": predictions

    }


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

        result = predict_disease(sys.argv[1])

        print()

        print("=" * 60)

        print("Prediction")

        print("=" * 60)

        print()

        print("Top Prediction")

        print()

        print(result["top_prediction"])

        print()

        print("Top Predictions")

        print()

        for prediction in result["predictions"]:

            print(
                f"{prediction['disease']:<15}"
                f"{prediction['confidence']:.2f}%"
            )