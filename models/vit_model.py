import timm
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import MODEL_NAME, NUM_CLASSES


def create_model():

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True
    )

    in_features = model.head.in_features

    model.head = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model
def save_confusion_matrix(
    cm,
    class_names,
    save_path
):

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()