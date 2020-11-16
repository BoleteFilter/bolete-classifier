## Transfer Learn on new data
## inspired by https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import torch
import torch.optim as optim
import torch.nn as nn
import copy
from data_utils import BoleteDataset, get_data_from_splits
from sklearn.model_selection import StratifiedShuffleSplit


def get_loader(X, Y, batch_size, shuffle=True):
    dataset = BoleteDataset(X, Y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


def train_model(
    model, dataloaders, loss_fn, optimizer, num_epochs, show_every, device,
):
    iter_count = 0
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    scores = model(inputs)
                    outputs = torch.sigmoid(scores)
                    loss = loss_fn(outputs, labels)
                    preds = outputs > 0.5
                    print(preds)
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
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def cross_val(
    X_train,
    Y_train,
    model,
    optimizer,
    loss_fn,
    batch_size,
    num_epochs,
    show_every,
    device=torch.device("cpu"),
):
    cross_val_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cross_val_split.get_n_splits(X_train, Y_train)
    for train_index, val_index in cross_val_split.split(X_train, Y_train):
        X_train_cv, X_val_cv, Y_train_cv, Y_val_cv = get_data_from_splits(
            X_train, Y_train, train_index, val_index
        )
        train_loader = get_loader(X_train_cv, Y_train_cv, batch_size)
        val_loader = get_loader(X_val_cv, Y_val_cv, batch_size)
        dataloaders = {"train": train_loader, "val": val_loader}
        train_model(
            model=model,
            dataloaders=dataloaders,
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_epochs=num_epochs,
            show_every=show_every,
            device=device,
        )
