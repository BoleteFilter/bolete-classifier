from builtins import range
import os, json
import numpy as np
import h5py

BASE_DIR = "data_pipeline"


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


if __name__ == "__main__":
    load_bolete_data()
