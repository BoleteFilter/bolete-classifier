## Transfer Learn on new data
## inspired by https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import copy

from sklearn.model_selection import StratifiedShuffleSplit

from data_utils import BoleteDataset, get_data_from_splits


def get_loader(X, Y, batch_size, shuffle=True, transform=None):
    dataset = BoleteDataset(X, Y, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


def train_model(
    model,
    optimizer,
    dataloaders,
    loss_fn,
    pred_fn,
    num_epochs,
    show_every,
    device,
    dtype,
    phases=["train", "val"],
):
    print("Training model on: ", device)
    model.to(device, dtype=dtype)
    iter_count = 0
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=dtype)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    scores = model(inputs)

                    labels = labels.type_as(scores)

                    loss = loss_fn(scores, labels)
                    preds = pred_fn(scores)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if iter_count % show_every == 0:
                print("Iter: {}".format(iter_count))
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

            iter_count += 1
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(float(epoch_acc))
                val_loss_history.append(float(epoch_loss))
            else:
                train_acc_history.append(float(epoch_acc))
                train_loss_history.append(float(epoch_loss))

    model.load_state_dict(best_model_wts)
    return train_acc_history, train_loss_history, val_acc_history, val_loss_history


def cross_val(
    X_train,
    Y_train,
    y_train,
    model,
    optimizer,
    loss_fn,
    pred_fn,
    batch_size,
    num_epochs,
    show_every,
    folds=5,
    test_size=0.2,
    device=torch.device("cpu"),
    dtype=torch.float32,
    transform=None,
):
    print("CV model on: ", device)
    cross_val_split = StratifiedShuffleSplit(
        n_splits=folds, test_size=test_size, random_state=0
    )
    cross_val_split.get_n_splits(X_train, Y_train)
    fold = 1
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    model_wts = copy.deepcopy(model.state_dict())
    for train_index, val_index in cross_val_split.split(X_train, Y_train):
        (
            X_train_cv,
            X_val_cv,
            Y_train_cv,
            Y_val_cv,
            y_train_cv,
            y_val_cv,
        ) = get_data_from_splits(X_train, Y_train, y_train, train_index, val_index)

        train_loader = get_loader(X_train_cv, Y_train_cv, batch_size, transform)
        val_loader = get_loader(X_val_cv, Y_val_cv, batch_size, transform)
        dataloaders = {"train": train_loader, "val": val_loader}

        print("CV Fold: ", fold)

        model.load_state_dict(model_wts)
        train_acc, train_loss, val_acc, val_loss = train_model(
            model=model,
            optimizer=optimizer,
            dataloaders=dataloaders,
            loss_fn=loss_fn,
            pred_fn=pred_fn,
            num_epochs=num_epochs,
            show_every=show_every,
            device=device,
            dtype=dtype,
        )
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        fold += 1
    return history


def evaluate(
    X,  # images
    Y,  # output
    y,  # labels
    model,
    out_dim,
    pred_fn,
    device=torch.device("cpu"),
    dtype=torch.float32,
    transform=None,
):
    print("Evaluating model on: ", device)
    model.to(device, dtype=dtype)
    model.eval()
    loader = get_loader(X, Y, batch_size=1, transform=transform)

    outputs = np.zeros((y.shape[0], out_dim))

    yshape = Y.shape
    if len(Y.shape) < 2:
        yshape = (Y.shape[0], 1)

    y_pred = np.zeros(yshape)
    y_true = np.zeros(yshape)

    row = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, dtype=dtype)
        labels = labels.to(device)
        scores = model(inputs)

        pred = pred_fn(scores)
        with torch.no_grad():
            outputs[row, :] = scores.cpu().numpy()
            y_pred[row, :] = pred.cpu().numpy()
            y_true[row, :] = labels.cpu().numpy()
        row += 1

    return outputs, y_pred, y_true, y

