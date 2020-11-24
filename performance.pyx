from locale import getpreferredencoding

from numpy.lib.type_check import real
from data_utils import *
import numpy as np

# cython
from lookalikes import (
    species_from_feats,
    lookalikes,
    get_edibility,
    get_tau_edibility,
    get_num_species,
    get_feats,
)

#######################################################################################
# Performance from Evaluation
#######################################################################################


def compute_and_save_performance(name):
    scores, y_pred, y_true, y_labels = load_raw_eval_data(name)

    model_type = 0 if "characteristic" in name else 1 if "direct" in name else ""

    ps, ps_ed = get_ps(scores, y_pred, y_labels, model_type)

    save_performance_data(np.arange(0, 101) / 100, ps, name)
    save_performance_data(np.arange(0, 101) / 100, ps_ed, name + "_ed")

    return True


def get_ps(scores, y_pred, y_labels, model_type):
    ps = np.zeros((101, len(y_labels)))
    ps_ed = np.zeros((101, len(y_labels)))
    for p in range(101):
        for i in range(len(y_labels)):
            ps[p, i] = get_performance_for_p(p, i, scores, y_pred, y_labels, model_type)
            ps_ed[p, i] = get_performance_for_p(
                p, i, scores, y_pred, y_labels, model_type, edibility=True
            )
    ps = np.mean(ps, axis=1)
    ps_ed = np.mean(ps_ed, axis=1)
    return ps, ps_ed


def get_performance_for_p(p, i, scores, y_pred, y_labels, model_type, edibility=False):
    p = p / 100
    perf = 0

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

    t = y_labels[i]
    if not edibility:
        perf = performance(tau_hat, t)
    else:
        perf = edibility_performance(tau_hat, y_pred[i], t, model_type)

    return perf


def performance(tau_hat, t):
    acc = t in tau_hat
    M = get_num_species()
    return ((M - len(tau_hat)) * acc) / M


def edibility_performance(tau_hat, y_pred, t, model_type):
    real_ed = get_edibility(t)
    if model_type == 0: ## characteristic
        tau_hat_ed = get_tau_edibility(tau_hat)
        pred_ed = (  # mode predicted edibility
            np.argmax(np.bincount(tau_hat_ed)) if len(tau_hat_ed) > 0 else -1
        )
        return 1 if real_ed == pred_ed else 0
    else: ## direct edibility
        return y_pred == real_ed

#######################################################################################
# Random Characteristic Performance
#######################################################################################


def random_char_performance(num_samples, type):
    name = "random_char"

    if type == "species":
        ps = get_random_performance(num_samples, edibility=False)
        ps = np.mean(ps, 2)
        ps = np.mean(ps, 1)
        save_performance_data(np.arange(0, 101) / 100, ps, name)

        print("species done")
        return True

    if type == "edibility":
        ps_ed = get_random_performance(num_samples, edibility=True)
        ps_ed = np.mean(ps_ed, 2)
        ps_ed = np.mean(ps_ed, 1)
        save_performance_data(np.arange(0, 101) / 100, ps_ed, name + "_ed")

        print("edibility done")

        return True


def get_random_performance(num_samples, edibility):
    C = 38
    results = np.zeros((101, num_samples, get_num_species()))

    U = np.random.randint(0, 2, (101, num_samples, C))

    for p in range(101):
        print("=", end="")
        for s in range(num_samples):
            u = U[p, s, :]  # characteristic
            for t in range(get_num_species()):  # target species
                results[p, s, t] = get_performance(u, t, p / 100, edibility)
    return results


def get_performance(u, t, p, edibility):
    tau_hat = species_from_feats(u, p)

    if edibility:
        real_ed = get_edibility(t)
        tau_hat_ed = get_tau_edibility(tau_hat)
        pred_ed = (  # mode predicted edibility
            np.argmax(np.bincount(tau_hat_ed)) if len(tau_hat_ed) > 0 else -1
        )
        res = 1 if real_ed == pred_ed else 0
    else:
        acc = t in tau_hat
        m = get_num_species()
        res = ((m - len(tau_hat)) * acc) / m
    return res


#######################################################################################
# Random Direct Performance
#######################################################################################


def random_direct_performance(num_samples, type):
    name = "random_direct"

    if type == "species":
        ps = get_random_direct_performance_species(num_samples)
        ps = np.mean(ps, 2)
        ps = np.mean(ps, 1)
        save_performance_data(np.arange(0, 101) / 100, ps, name)

        print("species done")
        return True

    if type == "edibility":
        ps_ed = get_random_direct_performance_ed(num_samples)
        ps_ed = np.mean(ps_ed) * np.ones((101,))
        save_performance_data(np.arange(0, 101) / 100, ps_ed, name + "_ed")
        return True


def get_random_direct_performance_ed(num_samples):
    M = get_num_species()
    res = np.zeros((M))
    for t in range(M):
        print("=", end="")
        pred_ed = np.random.randint(0, 4, (num_samples,))
        real_ed = get_edibility(t) * np.ones((num_samples,))
        res[t] = np.mean(pred_ed == real_ed)
    return res


def get_random_direct_performance_species(num_samples):
    M = get_num_species()
    results = np.zeros((101, num_samples, M))

    I = np.zeros((101, num_samples)).tolist()

    for row in range(101):
        for col in range(num_samples):
            I[row][col] = get_permuted_species_list(M)

    for p in range(101):
        print("=", end="")
        for s in range(num_samples):
            ids = I[p][s]
            for t in range(M):  # target species
                results[p, s, t] = get_direct_performance_species(ids, t, p / 100)
    return results


def get_permuted_species_list(M):
    return np.random.permutation(np.arange(M))


def get_direct_performance_species(ids, t, p):
    size_of_tau_hat = len(lookalikes(ids[0], p))
    tau_hat = ids[:size_of_tau_hat]

    acc = t in tau_hat
    m = get_num_species()
    res = ((m - len(tau_hat)) * acc) / m
    return res


#######################################################################################
# Perfect Species Performance
#######################################################################################


def perfect_performance(type):

    if type == "species":
        ps = get_perfect_performance(False)
        save_performance_data(np.arange(0, 101) / 100, ps, "perfect")
        return True

    if type == "edibility":
        ps_ed = get_perfect_performance(True)
        save_performance_data(np.arange(0, 101) / 100, ps_ed, "perfect_ed")

        return True


def get_perfect_performance(edibility):
    results = np.zeros((101, get_num_species()))

    for p in range(101):
        print("=", end="")
        for t in range(get_num_species()):
            u = get_feats(t)
            results[p, t] = get_performance(u, t, p / 100, edibility)
    print("done")
    results = np.mean(results, axis=1)
    return results
