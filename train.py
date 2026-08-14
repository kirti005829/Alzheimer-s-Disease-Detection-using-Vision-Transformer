import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from evaluate import evaluate
from config.config import *
from models.vit_model import create_model
from utils.dataset import get_dataloaders
from utils.transforms import get_transforms
from utils.history import TrainingHistory
from utils.plot import plot_history
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def main():

    set_seed(RANDOM_SEED)

    train_transform, test_transform = get_transforms()

    train_loader, test_loader, class_names = get_dataloaders(
        train_transform,
        test_transform
    )

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()
    history = TrainingHistory()
    start_epoch = 0

    best_accuracy = 0
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"\nUsing Device : {device}")
    print(f"Classes : {class_names}")
    print()

    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )
    
        val_loss, metrics = evaluate(
            model,
            test_loader,
            criterion,
            device
        )
    
        print()
    
        print(f"Train Loss : {train_loss:.4f}")
    
        print(f"Validation Loss : {val_loss:.4f}")
    
        print(f"Accuracy : {metrics['accuracy']:.4f}")
    
        print(f"Precision : {metrics['precision']:.4f}")
    
        print(f"Recall : {metrics['recall']:.4f}")
    
        print(f"F1 Score : {metrics['f1']:.4f}")
        history.update(
         epoch + 1,
         train_loss,
         val_loss,
         metrics["accuracy"],
         metrics["precision"],
         metrics["recall"],
         metrics["f1"])
        if metrics["accuracy"] > best_accuracy:
    
            best_accuracy = metrics["accuracy"]
    
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_accuracy,
                "checkpoints/best_model.pth")        
        
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_accuracy,
                "checkpoints/last_checkpoint.pth"
            )
    
            print()
    
            print("Best Model Saved!")

    history.save("results/training_history.csv")
    
    plot_history("results/training_history.csv")
    
    print("\nTraining Finished")
    print("Training history saved!")
    print("Graphs saved!")


if __name__ == "__main__":
    main()