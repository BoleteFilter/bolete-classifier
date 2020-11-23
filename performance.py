from locale import getpreferredencoding
from data_utils import *
from lookalikes import *
import torch
import numpy as np
from deco import *

#######################################################################################


def compute_and_save_performance(name):
    scores, y_pred, y_true, y_labels = load_raw_eval_data(name)

    model_type = 0 if "characteristic" in name else 1 if "direct" in name else ""

    ps, ps_ed = get_ps(scores, y_pred, y_labels, model_type)

    save_performance_data(np.arange(0, 101) / 100, ps, name)
    save_performance_data(np.arange(0, 101) / 100, ps_ed, name + "_ed")

    return True


@synchronized
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


@concurrent
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
        perf = edibility_performance(tau_hat, t)

    return perf


def performance(tau_hat, t):
    acc = t in tau_hat
    M = get_num_species()
    return ((M - len(tau_hat)) * acc) / M


def edibility_performance(tau_hat, t):
    real_ed = get_edibility(t)
    tau_hat_ed = get_tau_edibility(tau_hat)
    pred_ed = (  # mode predicted edibility
        np.argmax(np.bincount(tau_hat_ed)) if len(tau_hat_ed) > 0 else -1
    )
    return 1 if real_ed == pred_ed else 0


#######################################################################################


def random_char_performance():
    name = "random_char"
    num_samples = 100

    ps = get_random_performance(num_samples, edibility=False)
    ps = np.mean(ps, 2)
    ps = np.mean(ps, 1)
    save_performance_data(np.arange(0, 101) / 100, ps, name)

    print("species done")

    ps_ed = get_random_performance(num_samples, edibility=True)
    ps_ed = np.mean(ps_ed, 1)
    ps_ed = np.mean(ps_ed, 1)
    save_performance_data(np.arange(0, 101) / 100, ps_ed, name + "_ed")

    print("edibility done")

    return True


@synchronized
def get_random_performance(num_samples, edibility):
    C = 38
    results = np.zeros((101, num_samples, get_num_species()))

    U = np.random.randint(0, 2, (101, num_samples, C))

    for p in range(101):
        for s in range(num_samples):
            u = U[p, s, :]  # characteristic
            for t in range(get_num_species()):  # target species
                results[p, s, t] = get_performance(u, t, p / 100, edibility)
    return results


@concurrent
def get_performance(u, t, p, edibility):
    tau_hat = species_from_feats(u, p)
    # print(u)

    if edibility:
        real_ed = get_edibility(t)
        tau_hat_ed = get_tau_edibility(tau_hat)
        # print("tau_hat_ed = ", tau_hat_ed)
        pred_ed = (  # mode predicted edibility
            np.argmax(np.bincount(tau_hat_ed)) if len(tau_hat_ed) > 0 else -1
        )
        # print(pred_ed)
        res = 1 if real_ed == pred_ed else 0
    else:
        acc = t in tau_hat
        m = get_num_species()
        res = ((m - len(tau_hat)) * acc) / m
    # print(res)
    return res


#######################################################################################


def random_direct_performance():
    name = "random_direct"
    num_samples = 5

    # ps = get_random_direct_performance(num_samples, edibility=False)
    # ps = np.mean(ps, 2)
    # ps = np.mean(ps, 1)
    # save_performance_data(np.arange(0, 101) / 100, ps, name)

    # print("species done")

    ps_ed = get_random_direct_performance(num_samples, edibility=True)
    ps_ed = np.mean(ps_ed, 1)
    ps_ed = np.mean(ps_ed, 1)
    save_performance_data(np.arange(0, 101) / 100, ps_ed, name + "_ed")

    print("edibility done")

    return True


@synchronized
def get_random_direct_performance(num_samples, edibility):
    M = get_num_species()
    results = np.zeros((101, num_samples, M))

    I = np.zeros((101, num_samples)).tolist()

    for row in range(101):
        for col in range(num_samples):
            I[row][col] = get_permuted_species_list(M)

    for p in range(101):
        for s in range(num_samples):
            ids = I[p][i]  # characteristic
            for t in range(M):  # target species
                results[p, s, t] = get_direct_performance(ids, t, p / 100, edibility)
    return results


@concurrent
def get_permuted_species_list(M):
    return np.random.permutation(np.arange(M))


@concurrent
def get_direct_performance(ids, t, p, edibility):
    size_of_tau_hat = len(lookalikes(ids[0], p))
    tau_hat = ids[:size_of_tau_hat]

    if edibility:
        real_ed = get_edibility(t)
        tau_hat_ed = get_tau_edibility(tau_hat)
        # print("tau_hat_ed = ", tau_hat_ed)
        pred_ed = (  # mode predicted edibility
            np.argmax(np.bincount(tau_hat_ed)) if len(tau_hat_ed) > 0 else -1
        )
        # print(pred_ed)
        res = 1 if real_ed == pred_ed else 0
    else:
        acc = t in tau_hat
        m = get_num_species()
        res = ((m - len(tau_hat)) * acc) / m
    # print(res)
    return res


#######################################################################################


def get_performance_direct(ids, t, p, edibility=False):
    size_of_tau_hat = len(lookalikes(ids[0], p))
    tau_hat = ids[:size_of_tau_hat]
    if edibility:
        edibility_performance(tau_hat, t)
    else:
        performance(tau_hat, t)


def estimate_performance(u, p, edibility=False):
    res = np.zeros((get_num_species(),))
    for t in range(get_num_species()):
        perf = get_performance(u, t, p, edibility)
        # print(perf)
        res[t] = perf
    return res


def estimate_performance_direct(ids, p, edibility=False):
    res = np.zeros((get_num_species(),))
    for t in range(get_num_species()):
        perf = get_performance_direct(ids, t, p, edibility)
        res[t] = perf
    return res


if __name__ == "__main__":
    random_char_performance()
