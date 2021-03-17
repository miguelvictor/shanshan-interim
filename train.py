from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from datasets import ShanshanDataset
from models import SimGNN
from parsers import parameter_parser

import numpy as np
import pytorch_lightning as pl
import torch


def format_confusion_matrix(cm):
    x = PrettyTable()
    x.field_names = ["", "Model (-)", "Model (+)"]
    x.add_row(["Actual (-)", cm[0][0], cm[0][1]])
    x.add_row(["Actual (+)", cm[1][0], cm[1][1]])
    return x


def train(epochs=3):
    # parse CLI args
    args = parameter_parser()

    # load training and validation data
    train_ds = ShanshanDataset(key='train')
    train_dl = DataLoader(train_ds, batch_size=1)
    val_ds = ShanshanDataset(key='dev')
    val_dl = DataLoader(val_ds, batch_size=1)

    # load model and trainer
    model = SimGNN(args, train_ds.n_labels)

    # start training (model with the lowest validation loss is saved)
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    trainer = pl.Trainer(gpus=gpus, max_epochs=epochs)
    trainer.fit(model, train_dl, val_dl)

    # evaluate on the testing set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in ShanshanDataset(key='test'):
            target = data.pop('target')
            prediction = torch.sigmoid(model(data)).squeeze()
            y_true.append(target)
            y_pred.append(prediction)

        # calculate statistics
        cm = confusion_matrix(y_true, np.around(y_pred))
        acc = accuracy_score(y_true, np.around(y_pred))
        score = roc_auc_score(y_true, y_pred)
        report = classification_report(y_true, np.around(y_pred))

        print('=' * 60)
        print(f'Accuracy: {acc:.4%}')
        print(f'ROC-AUC Score: {score:.4%}')
        print(format_confusion_matrix(cm))
        print(report)
        print('=' * 60)


if __name__ == '__main__':
    train()
