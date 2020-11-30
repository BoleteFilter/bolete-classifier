import numpy as np
import pandas as pd

features = np.loadtxt("./data_pipeline/clean_data/new_features.csv", delimiter=",")
edibility = np.loadtxt("./data_pipeline/clean_data/new_edibility.csv", delimiter=",")
names = pd.read_csv("./data_pipeline/clean_data/final_ids.csv")


def get_num_species():
    return edibility.shape[0]


def get_feats(id):
    return features[id]


def get_name(id):
    return names.loc[names["NewID"] == id]["NewName"].to_string().split()[1]


def get_edibility(id):
    ed = np.argmax(edibility[id])
    return ed


def get_tau_edibility(tau_hat):
    return np.argmax(edibility[tau_hat], axis=1)


def similarity(a, b):
    return np.sum(a * b)


def lookalikes(spec_id, p):
    u = get_feats(spec_id)
    idxs = np.where(features.dot(u.T) >= p * similarity(u, u))[0]
    return idxs


def species_from_feats(u, p):
    return np.where(features.dot(u.T) >= p * similarity(u, u))[0]


def print_feats(species_id):
    feats = get_feats(species_id)
    idxs = np.where(feats > 0)[1]
    print(idxs)
    for idx in idxs:
        print(features[idx])
    print()

