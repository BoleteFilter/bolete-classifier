import pandas as pd
import numpy as np
from collections import defaultdict

labels = pd.read_csv("./clean_data/labels.csv")
features = pd.read_csv("./clean_data/features.csv")

feats = [col for col in labels if col.startswith("F")][:-5]


def get_feats(species_id):
    features = labels.loc[labels["ID"] == species_id][feats]
    return features.to_numpy()


def get_name(species_id):
    return labels.loc[labels["ID"] == species_id]["NewName"].to_string().split()[1]


def similarity(a, b):
    return np.sum(a * b)


spec_id = 38
p = 0.7

## Vectorized Version ##
Feats4All = labels.loc[:, feats].to_numpy()


def lookalikes_v(spec_id, p):
    species = get_feats(spec_id)
    idxs = np.where(Feats4All.dot(species.T) >= p * similarity(species, species))[0]
    return idxs


idxs = lookalikes_v(spec_id, p)
print(get_name(spec_id))
print([get_name(idx + 1) for idx in idxs])


## Iterative version


# def lookalikes_i(species_id, p):
#     species = get_feats(species_id)
#     sims = defaultdict(int)
#     for spec_id in labels["ID"]:
#         if spec_id is not species_id:
#             other_species = get_feats(spec_id)
#             sim = similarity(species, other_species)
#             sims[spec_id] = sim
#     result = [
#         spec for (spec, sim) in sims.items() if sim >= p * similarity(species, species)
#     ]
#     return result


# results = lookalikes_i(spec_id, p)
# print()
# print(spec_id, ", ", get_name(spec_id))
# for result in results:
#     print(get_name(result))

pd.set_option("display.max_colwidth", 100)


def print_feats(species):
    feats = get_feats(species)
    idxs = np.where(feats > 0)[1]
    print(idxs)
    for idx in idxs:
        print(features.loc[features["id"] == idx + 1]["feature"].to_string()[-15:])
    print()
