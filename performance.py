from data_utils import *
from lookalikes import *
import torch
import numpy as np

def shared(tau, tau_hat):
    return len(np.intersect1d(tau, tau_hat))

def compute_and_save_performance(name):
    scores, y_pred, y_true, y_labels = load_raw_eval_data(name)

    model_type = 0 if "characteristic" in name else 1 if "direct" in name else ""

    ps = {}
    ps_ed = {}
    for p in range(0, 100):
        p = p / 100
        ps[p] = []
        ps_ed[p] = 0
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
            true_lookalikes = lookalikes(t, p)
            tau_overlap = shared(tau_hat, true_lookalikes)/max(len(true_lookalikes), len(tau_hat))
            ps[p].append(performance(tau_hat, t))
            real_ed = get_edibility(t)
            pred_ed = np.argmax(np.bincount(tau_hat)) if len(tau_hat) > 0 else -1
            ps_ed[p] += 1 if real_ed == pred_ed else 0

        ps[p] = np.mean(ps[p])
        ps_ed[p] = ps_ed[p] / len(y_labels)

    keys = np.array(list(ps.keys()))
    values = np.array(list(ps.values()))
    save_performance_data(keys, values, name)

    ed_keys = np.array(list(ps_ed.keys()))
    ed_values = np.array(list(ps_ed.values()))
    save_performance_data(ed_keys, ed_values, name + "_ed", edibility=False)
    return True