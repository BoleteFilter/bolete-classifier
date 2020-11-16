from builtins import range
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import os, json
import numpy as np
import h5py

BASE_DIR = "data_pipeline"


class BoleteDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]
        if self.transform:
            image = self.transform(image)
        return (image, label)


def load_bolete_data(base_dir=BASE_DIR, max_train=None):
    data = {}
    bolete_file = os.path.join(base_dir, "bolete.h5")
    with h5py.File(bolete_file, "r") as f:
        print(f.items())
        for k, v in f.items():
            data[k] = np.asarray(v)

    # # # Maybe subsample the training data
    # if max_train is not None:
    #     num_train = data["train_captions"].shape[0]
    #     mask = np.random.randint(num_train, size=max_train)
    #     data["train_captions"] = data["train_captions"][mask]
    #     data["train_image_idxs"] = data["train_image_idxs"][mask]

    return data


def get_train_and_test(data, y_type):
    X, y = data["bolete-images"].T, data["bolete-labels"].T
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test, Y_train, Y_test = get_data_from_splits(
            X, data[y_type].T, train_index, test_index
        )
    return X_train, X_test, Y_train, Y_test


def get_data_from_splits(X, Y, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    load_bolete_data()
