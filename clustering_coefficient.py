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
    trials=100
) -> float:
    # assert state != []
    # assert trials < len(state)

    n = len(state)

    closed_triang = 0
    for trial in [int(random.random() * n) for _ in range(trials)]:
        fst_ord_connections = [x for x in nbs[trial] if state[x] == state[trial] and x != -1]
        if len(fst_ord_connections) < 2:
            continue
        u, v = random.sample(fst_ord_connections, 2)
        if v in nbs[u]:
            closed_triang += 1

    return closed_triang / trials

def approx_global_coefficients_over_time(
    states,
    nbs_filtered,
    interval=10,
    trials=200
) -> list[float]:
    res = []
    for state_i in tqdm(range(0, len(states), interval), position=1):
        res.append(approx_global_coefficient(states[state_i], nbs_filtered, trials))
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
    return [get_local_coefficients(states[state_i], nbs) for state_i in tqdm(range(0, len(states), interval), position=1)]





# def test_loc


def main():
    pass
    # adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
    #     num_nodes=500, edge_prob=0.4
    # )
    # adj = np.asarray([
    #     [0, 0, 1, 0],
    #     [0, 0, 1, 1],
    #     [1, 1, 0, 1],
    #     [0, 1, 1, 0],
    #     ])
    # from sparse_adj_mat import Sparse_Adjacency_Matrix
    # adj_mat = Sparse_Adjacency_Matrix(adj)


    # random_init = jax.random.randint(
    #     shape=(len(adj_mat),),
    #     minval=-1,
    #     maxval=2,
    #     key=jax.random.PRNGKey(np.random.randint(low=0, high=100)),
    # )
    # ones_init = jnp.ones(shape=(len(adj_mat),)).astype(int)

    # network = ising_efficient.BeliefNetwork(
    #     sparse_adj=adj_mat,
    #     external_field=lambda t, node_idx: jnp.sin(t * 0.1),
    #     init_state=random_init,
    #     µ=0.9,
    #     beta=0.5,
    # )

    # result = network.run_for_steps(1000)
    # all_states = [np.array(res) for res in result]

    # # coefficients = get_local_coefficients_over_time(
    # #     sparse_adj=adj_mat,
    # #     states=all_states,
    # #     interval=40
    # # )

    # # mean_local_coeff = [np.mean(list(coefficients[i].values())) for i in coefficients]
    # # plt.plot(coefficients.keys(), mean_local_coeff)
    # # plt.show()

    # glob_coeff_over_time = approx_global_coefficients_over_time(
    #     sparse_adj=adj_mat,
    #     states=all_states,
    #     interval=200,
    #     trials=200
    # )


    # print(glob_coeff_over_time)

if __name__ == "__main__":
    main()






