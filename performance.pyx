from locale import getpreferredencoding
from data_utils import *
import numpy as np

# cython
from lookalikes import (
    species_from_feats,
    lookalikes,
    get_edibility,
    get_tau_edibility,
    get_num_species,
)

#######################################################################################
# Random Characteristic Performance
#######################################################################################


def random_char_performance(num_samples):
    name = "random_char"

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
# Random Direct Performance
#######################################################################################


def random_direct_performance(num_samples):
    name = "random_direct"

    ps = get_random_direct_performance(num_samples, edibility=False)
    ps = np.mean(ps, 2)
    ps = np.mean(ps, 1)
    save_performance_data(np.arange(0, 101) / 100, ps, name)

    print("species done")

    ps_ed = get_random_direct_performance(num_samples, edibility=True)
    ps_ed = np.mean(ps_ed, 1)
    ps_ed = np.mean(ps_ed, 1)
    save_performance_data(np.arange(0, 101) / 100, ps_ed, name + "_ed")

    print("edibility done")

    return True


def get_random_direct_performance(num_samples, edibility):
    M = get_num_species()
    results = np.zeros((101, num_samples, M))

    I = np.zeros((101, num_samples)).tolist()

    for row in range(101):
        for col in range(num_samples):
            I[row][col] = get_permuted_species_list(M)

    for p in range(101):
        print("=", end="")
        for s in range(num_samples):
            ids = I[p][s]  # characteristic
            for t in range(M):  # target species
                results[p, s, t] = get_direct_performance(ids, t, p / 100, edibility)
    return results


def get_permuted_species_list(M):
    return np.random.permutation(np.arange(M))


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

