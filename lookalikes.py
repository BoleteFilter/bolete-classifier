import pandas as pd
import numpy as np
from collections import defaultdict

features = pd.read_csv('./clean_data/features.csv')
labels = pd.read_csv('./clean_data/labels.csv')

feats = [col for col in labels if col.startswith("F")]

def get_feats(species_mid):
    features = labels.loc[labels["mId"] == species_mid][feats]
    return features.to_numpy()

def get_name(species_mid):
    return labels.loc[labels["mId"] == species_mid]['NewName'].to_string()

def distance(a, b):
    return np.abs(np.sum(a-b))

def lookalikes(species_mid, p):
    species = get_feats(species_mid)
    dists = defaultdict(int)
    for spec_mid in labels['mId']:
        if spec_mid is not species_mid:
            other_species = get_feats(spec_mid)
            dist = distance(species, other_species)
            dists[spec_mid] = dist
    l = (100.0001-p)/100*len(feats)
    print(l)
    result = [spec for (spec,dist) in dists.items() if dist <= l]
    return result

results = lookalikes(313, 100)
for result in results:
    print(get_name(result))


## Vectorized Version ##