import torch
from PIL import Image

from models.vit_model import create_model
from utils.transforms import get_transforms
from config.config import DEVICE


CLASS_NAMES = ["AD", "CI", "CN"]


model = create_model()

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

_, test_transform = get_transforms()


def predict(image):

    image = image.convert("RGB")

    image = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probability = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(probability, dim=1)

    return {
        "prediction": CLASS_NAMES[prediction.item()],
        "confidence": round(confidence.item() * 100, 2)
    }