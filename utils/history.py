import pandas as pd


class TrainingHistory:

    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

    def update(
        self,
        epoch,
        train_loss,
        val_loss,
        accuracy,
        precision,
        recall,
        f1
    ):

        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["accuracy"].append(accuracy)
        self.history["precision"].append(precision)
        self.history["recall"].append(recall)
        self.history["f1"].append(f1)

    def save(self, path):

        df = pd.DataFrame(self.history)

        df.to_csv(path, index=False)