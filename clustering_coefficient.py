import ising_efficient
import network_generation
from sparse_adj_mat import Sparse_Adjacency_Matrix
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def approx_global_coefficient(
    state,
    nbs,
    n_samples=100
) -> float:
    """
    Approimates global clustering coefficient by calculating fraction of closed 
    triangles over open triangles
    """
    n = len(state)

    closed_triang = 0
    for sample in [int(random.random() * n) for _ in range(n_samples)]:
        # collect all neighbours of same belief
        fst_ord_connections = [x for x in nbs[sample] if state[x] == state[sample] and x != -1]
        # no closed triangle if sample node doesn't have enough neighbours of similar belief
        if len(fst_ord_connections) < 2:
            continue
        # sample two same-belief neighbours
        u, v = random.sample(fst_ord_connections, 2)
        # check if neighbours are connected
        if v in nbs[u]:
            closed_triang += 1

    return closed_triang / n_samples


def approx_global_coefficients_over_time(
    states,
    nbs_filtered,
    interval=10,
    n_samples=100
) -> list[float]:
    """
    returns the approximated global clustering coefficient over time, with interval
    """
    res = []
    for state_i in tqdm(range(0, len(states), interval), position=1):
        res.append(approx_global_coefficient(states[state_i], nbs_filtered, n_samples))
    return res


def get_local_coefficients(
    state,
    nbs
) -> list[float]:
    results = []
    nbs = [np.asarray(nb[nb > -1]) for nb in nbs]
    for node_i, belief in enumerate(np.asarray(state)):
        poss_connections = (len(nbs[node_i]) * (len(nbs[node_i]) - 1)) / 2

        if poss_connections == 0:
            results.append(0)
            continue

        first_ord_conn = nbs[node_i]
        sec_ord_connections = []
        for nb in first_ord_conn:
            if state[nb] != belief:
                continue

            for edge in nbs[nb]:
                if edge == node_i \
                        or edge not in first_ord_conn \
                        or set([nb, edge]) in sec_ord_connections:
                    continue

                sec_ord_connections.append(set([nb, edge]))
        results.append(len(sec_ord_connections) / poss_connections)
    return np.array(results)


def get_local_coefficients_over_time(
    states,
    nbs,
    interval = 1
) -> list[list[float]]:
    """
    returns the approximated local clustering coefficients over time, with interval
    """
    return [get_local_coefficients(states[state_i], nbs) for state_i in tqdm(range(0, len(states), interval), position=1)]
