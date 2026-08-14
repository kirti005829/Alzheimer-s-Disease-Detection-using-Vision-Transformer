import argparse
from pathlib import Path

import torch
from PIL import Image

from config.config import *
from models.vit_model import create_model
from utils.transforms import get_transforms


CLASS_NAMES = ["AD", "CI", "CN"]


def load_model():

    model = create_model()

    checkpoint = torch.load(
        "checkpoints/best_model.pth",
        map_location="cpu"
    )

    # Supports both checkpoint dictionary and plain state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def predict(image_path):

    _, test_transform = get_transforms()

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0)

    model = load_model()

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    return (
        predicted.item(),
        confidence.item(),
        probabilities.squeeze().tolist()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path of MRI image"
    )

    args = parser.parse_args()

    pred, conf, probs = predict(args.image)

    print("\n========== Prediction ==========\n")

    print(
        f"Predicted Class : {CLASS_NAMES[pred]}"
    )

    print(
        f"Confidence      : {conf*100:.2f}%"
    )

    print("\nProbabilities\n")

    for name, p in zip(CLASS_NAMES, probs):

        print(
            f"{name:3} : {p*100:.2f}%"
        )


if __name__ == "__main__":
    main()