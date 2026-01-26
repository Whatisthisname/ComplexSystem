import ising_efficient
import network_generation
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import random

def approx_global_coefficient(
    state,
    nbs,
    interval = 100,
    trials = 10
):
    n = len(state)
    assert (trials < n)

    closed_triang = 0
    for trial in [int(random.random() * n) for _ in range(trials)]:
        sec_ord_connections = [x for x in nbs[trial] if state[x] == state[trial]]
        if len(sec_ord_connections) < 2:
            continue
        u, v = random.sample(sec_ord_connections, 2)
        if v in nbs[u]:
            closed_triang += 1

    return closed_triang / n

def approx_global_coefficients_over_time(
    sparse_adj,
    states,
    interval = 100,
    trials = 200
):
    nbs, _ = sparse_adj.precompute_neighbors_and_weights()
    nbs = [np.asarray(nb[nb > -1]) for nb in nbs]

    results = {}
    for state_i in range(0, len(states), interval):
        results[state_i] = approx_global_coefficient(states[state_i], nbs)
    return results

def get_local_coefficients(
    state,
    nbs
) -> dict[int, float]:
    results = {}
    nbs = [np.asarray(nb[nb > -1]) for nb in nbs]
    for node_i, belief in enumerate(np.asarray(state)):
        poss_connections = (len(nbs[node_i]) * (len(nbs[node_i]) - 1)) / 2

        if poss_connections == 0:
            results[node_i] = 0
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
        results[node_i] = len(sec_ord_connections) / poss_connections
    return results

def get_local_coefficients_over_time(
    sparse_adj:np.ndarray,
    states:np.ndarray,
    interval:int = 100
) -> dict[int, dict[int, float]]:
    n_states = len(states)
    nbs, _ = sparse_adj.precompute_neighbors_and_weights()
    return {state_i: get_local_coefficients(states[state_i], nbs) for state_i in range(0, n_states, interval)}


adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
    num_nodes=500, edge_prob=0.4
)

# adj = np.asarray([
#     [0, 0, 1, 0],
#     [0, 0, 1, 1],
#     [1, 1, 0, 1],
#     [0, 1, 1, 0],
#     ])
# from sparse_adj_mat import Sparse_Adjacency_Matrix
# adj_mat = Sparse_Adjacency_Matrix(adj)


random_init = jax.random.randint(
    shape=(len(adj_mat),),
    minval=-1,
    maxval=2,
    key=jax.random.PRNGKey(np.random.randint(low=0, high=100)),
)
ones_init = jnp.ones(shape=(len(adj_mat),)).astype(int)

network = ising_efficient.BeliefNetwork(
    sparse_adj=adj_mat,
    external_field=lambda t, node_idx: jnp.sin(t * 0.1),
    init_state=random_init,
    µ=0.9,
    beta=0.5,
)

result = network.run_for_steps(1000)
all_states = [np.array(res) for res in result]

# coefficients = get_local_coefficients_over_time(
#     sparse_adj=adj_mat,
#     states=all_states,
#     interval=40
# )

# mean_local_coeff = [np.mean(list(coefficients[i].values())) for i in coefficients]
# plt.plot(coefficients.keys(), mean_local_coeff)
# plt.show()

glob_coeff_over_time = approx_global_coefficients_over_time(
    sparse_adj=adj_mat,
    states=all_states,
    interval=200,
    trials=200
)


print(glob_coeff_over_time)






