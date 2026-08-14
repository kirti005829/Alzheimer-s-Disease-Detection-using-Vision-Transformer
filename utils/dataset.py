from torchvision import datasets
from torch.utils.data import DataLoader

from config.config import *


def get_dataloaders(train_transform, test_transform):

    train_dataset = datasets.ImageFolder(
        root=DATA_DIR / "dataset" / "train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=DATA_DIR / "dataset" / "test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    class_names = train_dataset.classes

    return train_loader, test_loader, class_names