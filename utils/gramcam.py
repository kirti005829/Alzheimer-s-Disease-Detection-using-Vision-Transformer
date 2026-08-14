import cv2
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.vit_model import create_model
from utils.transforms import get_transforms


CLASS_NAMES = ["AD", "CI", "CN"]


def reshape_transform(tensor, height=14, width=14):
    """
    Convert ViT output tokens into CNN-like feature maps.

    Input shape:
        (B, 197, 768)

    Output shape:
        (B, 768, 14, 14)
    """

    tensor = tensor[:, 1:, :]        # Remove CLS token

    tensor = tensor.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2)
    )

    tensor = tensor.permute(
        0,
        3,
        1,
        2
    )

    return tensor
def load_model():

    model = create_model()

    checkpoint = torch.load(
        "checkpoints/best_model.pth",
        map_location="cpu"
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    return model
def generate_gradcam(image_path):

    model = load_model()

    target_layers = [
        model.blocks[-1].norm1
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    _, test_transform = get_transforms()

    image = Image.open(image_path).convert("RGB")

    rgb_image = np.array(image).astype(np.float32) / 255

    input_tensor = test_transform(image).unsqueeze(0)

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        "results/gradcam_output.png",
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    print("Grad-CAM saved!")
if __name__ == "__main__":

    generate_gradcam(
        "data/dataset/test/AD/YOUR_IMAGE_NAME.jpg"
    )