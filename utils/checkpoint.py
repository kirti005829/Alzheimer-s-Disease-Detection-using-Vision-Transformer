import torch
import os


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_accuracy,
    path
):

    checkpoint = {

        "epoch": epoch,

        "best_accuracy": best_accuracy,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict()

    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    path
):

    if not os.path.exists(path):

        return model, optimizer, 0, 0

    checkpoint = torch.load(path)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]

    best_accuracy = checkpoint["best_accuracy"]

    print(f"Checkpoint Loaded (Epoch {epoch})")

    return model, optimizer, epoch, best_accuracy