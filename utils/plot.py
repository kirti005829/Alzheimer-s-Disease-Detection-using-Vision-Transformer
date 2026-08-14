import matplotlib.pyplot as plt
import pandas as pd


def plot_history(csv_path):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8,5))

    plt.plot(df["epoch"], df["train_loss"], label="Train")

    plt.plot(df["epoch"], df["val_loss"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.savefig("results/loss_curve.png")

    plt.close()


    plt.figure(figsize=(8,5))

    plt.plot(df["epoch"], df["accuracy"])

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Accuracy")

    plt.savefig("results/accuracy_curve.png")

    plt.close()