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
    assert len(state) > 2
    assert len(nbs) > 0
    assert n_samples > 0

    # filter out all non-existing edges
    nbs = [np.asarray(nb[nb != -1]) for nb in nbs]

    closed_triang = 0
    for sample in [int(random.random() * len(state)) for _ in range(n_samples)]:
        # collect all neighbours of same belief
        fst_ord_connections = [x for x in nbs[sample] if state[x] == state[sample]]
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
    nbs,
    interval=10,
    n_samples=100
) -> list[float]:
    """
    Returns the approximated global clustering coefficient over time, with interval
    """
    assert len(states) > 2
    assert len(nbs) > 0
    assert interval > 0
    assert n_samples > 0

    res = []
    for state_i in tqdm(range(0, len(states), interval), position=1):
        res.append(approx_global_coefficient(states[state_i], nbs, n_samples))
    return res


def get_local_coefficients(
    state,
    nbs
) -> list[float]:
    """
    Calculates local clustering coefficient per node
    """
    assert len(state) > 2
    assert len(nbs) > 0

    # filter for non-existing neighbour nodes
    nbs = [np.asarray(nb[nb != -1]) for nb in nbs]

    results = []
    # per node and it's belief
    for node_i, belief in enumerate(np.asarray(state)):
        n_nbs = len(nbs[node_i])
        # max possible connections between node_i's neighbours
        poss_connections = n_nbs * (n_nbs - 1) / 2

        if poss_connections == 0:
            results.append(.0)
            continue

        # all node_i's neighbours, filtered on belief
        first_ord_connections = [x for x in nbs[node_i] if state[x] == belief]

        # store all inter-neighbour connections
        sec_ord_connections = []
        for foc in first_ord_connections:
            for soc in nbs[foc]:
                # don't store if:
                #     connection towards original node
                #     connection not a neighbour of first node
                #     connection already looked at from other neighbours side
                if soc == node_i \
                        or soc not in first_ord_connections \
                        or set([foc, soc]) in sec_ord_connections:
                    continue

                # otherwise: store edge
                sec_ord_connections.append(set([foc, soc]))
        results.append(len(sec_ord_connections) / poss_connections)
    return np.array(results)


def get_local_coefficients_over_time(
    states,
    nbs,
    interval=10
) -> list[list[float]]:
    """
    Returns the local clustering coefficients over time, with interval
    """
    assert len(states) > 2
    assert nbs > 0
    assert interval > 0
    assert n_samples > 0

    res = []
    for state_i in tqdm(range(0, len(states), interval), position=1):
        res.append(get_local_coefficients(states[state_i], nbs))
    return res
