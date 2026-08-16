import torch
from PIL import Image

from models.vit_model import create_model
from utils.transforms import get_transforms


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "AD",
    "CI",
    "CN"
]

MODEL_PATH = "checkpoints/best_model.pth"
def load_model():

    model = create_model()

    checkpoint = torch.load(MODEL_PATH)

    model.load_state_dict(
    checkpoint["model_state_dict"]
)

    model.to(DEVICE)

    model.eval()

    return model
def predict_image(image_path):

    model = load_model()

    _, test_transform = get_transforms()

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return (
        CLASS_NAMES[prediction.item()],
        confidence.item()
    )
if __name__ == "__main__":

    image_path = input(
        "Enter Image Path : "
    )

    prediction, confidence = predict_image(
        image_path
    )

    print()

    print(f"Prediction : {prediction}")

    print(
        f"Confidence : {confidence*100:.2f}%"
    )