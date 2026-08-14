from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted"
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted"
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted"
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred
        ),
        "classification_report": classification_report(
            y_true,
            y_pred
        )
    }

    return metrics


def save_confusion_matrix(
    cm,
    class_names,
    save_path
):

    plt.figure(figsize=(8, 6))

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