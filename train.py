from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from datasets import ShanshanDataset
from models import SimGNN
from parsers import parameter_parser

import pytorch_lightning as pl
import torch


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
            prediction = torch.round(model(data)).squeeze()
            y_true.append(target)
            y_pred.append(prediction)

        matrix = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix', matrix)


if __name__ == '__main__':
    train()
