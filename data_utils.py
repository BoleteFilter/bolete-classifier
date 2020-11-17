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

    return data


def get_data_from_splits(X, Y, y, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, Y_train, Y_test, y_train, y_test


def get_train_and_test(data, y_type):
    X, y = data["bolete-images"].T, data["bolete-labels"].T
    X = np.transpose(X, [0, 3, 1, 2])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test, Y_train, Y_test, y_train, y_test = get_data_from_splits(
            X, data[y_type].T, data["bolete-labels"].T, train_index, test_index
        )
    return X_train, X_test, Y_train, Y_test, y_train, y_test


def get_val(X_train, Y_train, y_train):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    sss.get_n_splits(X_train, y_train)
    for train_index, test_index in sss.split(X_train, y_train):
        _, X, _, Y, _, y = get_data_from_splits(
            X_train, Y_train, y_train, train_index, test_index
        )
    return X, Y, y


def save_raw_eval_data(scores, y_pred, y_true, y_labels, name):
    np.savetxt(
        "evaluation_data/" + name + "_scores.csv", y_pred, delimiter=",", fmt="%d"
    )
    np.savetxt("evaluation_data/" + name + "_pred.csv", y_pred, delimiter=",", fmt="%d")
    np.savetxt("evaluation_data/" + name + "_true.csv", y_true, delimiter=",", fmt="%d")
    np.savetxt(
        "evaluation_data/" + name + "_label.csv", y_labels, delimiter=",", fmt="%d"
    )
    return True


def load_raw_eval_data(name):
    np.loadtxt(
        "evaluation_data/" + name + "_scores.csv", delimiter=",", dtype=float,
    )
    np.loadtxt(
        "evaluation_data/" + name + "_pred.csv", delimiter=",", dtype=int,
    )
    np.loadtxt(
        "evaluation_data/" + name + "_true.csv", delimiter=",", dtype=int,
    )
    np.loadtxt(
        "evaluation_data/" + name + "_label.csv", delimiter=",", dtype=int,
    )
    return True


def save_performance_data(p_value, perf, name):
    assert p_value.shape == perf.shape
    np.savetxt(
        "evaluation_data/" + name + "_perform.csv",
        [p_value, perf],
        delimiter=",",
        fmt="%d",
    )
    return True


def load_performance_data(p_value, perf, name):
    assert p_value.shape == perf.shape
    np.savetxt(
        "evaluation_data/" + name + "_perform.csv",
        [p_value, perf],
        delimiter=",",
        fmt="%d",
    )
    return True
