import pandas as pd
import numpy as np
from collections import defaultdict

labels = pd.read_csv("./clean_data/labels.csv")

feats = [col for col in labels if col.startswith("F")][:-5]

## Iterative version


def get_feats(species_id):
    features = labels.loc[labels["ID"] == species_id][feats]
    return features.to_numpy()


def get_name(species_id):
    return labels.loc[labels["ID"] == species_id]["NewName"].to_string().split()[1]


def similarity(a, b):
    return np.sum(a * b)


def lookalikes(species_id, p):
    species = get_feats(species_id)
    sims = defaultdict(int)
    for spec_id in labels["ID"]:
        if spec_id is not species_id:
            other_species = get_feats(spec_id)
            sim = similarity(species, other_species)
            sims[spec_id] = sim
    result = [
        spec for (spec, sim) in sims.items() if sim >= p * similarity(species, species)
    ]
    return result


spec_id = 250
results = lookalikes(spec_id, 0.9)
print(spec_id, ", ", get_name(spec_id))
for result in results:
    print(get_name(result))
print()


## Vectorized Version ##
p = 0.9
Feats4All = labels.loc[:, feats].to_numpy()
species = get_feats(spec_id)
print(get_name(spec_id))
idxs = np.where(Feats4All.dot(species.T) >= p * similarity(species, species))[0]
print([get_name(idx + 1) for idx in idxs])
