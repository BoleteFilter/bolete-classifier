from locale import getpreferredencoding
from data_utils import *
from lookalikes import *
import torch
import numpy as np
from deco import *


@concurrent
def get_performance_for_p(p, scores, y_pred, y_labels, model_type):
    p = p / 100
    ps = []
    ps_ed = 0
    for i in range(len(y_labels)):
        t = y_labels[i]

        if model_type == 0:  # characteristic
            tau_hat = species_from_feats(y_pred[i], p)

        elif model_type == 1:  # direct
            i_scores = list(enumerate(scores[i]))
            sorted_scores = sorted(i_scores, key=lambda x: x[1], reverse=True)
            size_of_tau_hat = len(lookalikes(sorted_scores[0][0], p))
            tau_hat = [idx for (idx, val) in sorted_scores[:size_of_tau_hat]]

        else:
            print("Unknown model type")
            return

        ps.append(performance(tau_hat, t))
        ps_ed += edibility_performance(tau_hat, t)

    ps = np.mean(ps)
    ps_ed = ps_ed / len(y_labels)
    return (ps, ps_ed)


@synchronized
def get_ps(scores, y_pred, y_labels, model_type):
    ps = {}
    for p in range(0, 100):
        ps[p] = get_performance_for_p(p, scores, y_pred, y_labels, model_type)
    return ps


def compute_and_save_performance(name):
    scores, y_pred, y_true, y_labels = load_raw_eval_data(name)

    model_type = 0 if "characteristic" in name else 1 if "direct" in name else ""

    ps = get_ps(scores, y_pred, y_labels, model_type)

    keys = np.array(list(ps.keys()))
    values = [a for a, _ in enumerate(ps.values())]
    ed_values = [b for _, b in enumerate(ps.values())]
    values = np.array(values)
    save_performance_data(keys, values, name)

    # ed_keys = np.array(list(ps_ed.keys()))
    ed_values = np.array(ed_values)
    save_performance_data(keys, ed_values, name + "_ed", edibility=False)
    return True
