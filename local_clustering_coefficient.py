import ising_efficient
import network_generation
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt


def get_local_coefficients(
    state,
    nbs
) -> dict[int, float]:
    results = {}
    nbs = [
        np.asarray(nb[nb > -1]) for nb in nbs
    ]
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
    all_states:np.ndarray,
    interval:int = 1
) -> dict[int, dict[int, float]]:
    n_states = len(all_states)
    nbs, _ = sparse_adj.precompute_neighbors_and_weights()
    return {state_i: get_local_coefficients(all_states[state_i], nbs) for state_i in range(0, n_states, interval)}


adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
    num_nodes=100, edge_prob=0.4
)

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

coefficients = get_local_coefficients_over_time(
    sparse_adj=adj_mat,
    all_states=all_states,
    interval=40
)

mean_local_coeff = [np.mean(list(coefficients[i].values())) for i in coefficients]
plt.plot(coefficients.keys(), mean_local_coeff)
plt.show()


