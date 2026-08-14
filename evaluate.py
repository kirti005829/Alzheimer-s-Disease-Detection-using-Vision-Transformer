import torch

from tqdm import tqdm

from utils.metrics import (calculate_metrics,save_confusion_matrix)


def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0

    predictions = []

    labels_list = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())

            labels_list.extend(labels.cpu().numpy())

    metrics = calculate_metrics(
        labels_list,
        predictions
    )

    val_loss = running_loss / len(loader)
    save_confusion_matrix(metrics["confusion_matrix"],
    ["AD","CI","CN"],"results/confusion_matrix.png")
    with open("results/classification_report.txt","w") as f:

        f.write(metrics["classification_report"])
    return val_loss, metrics