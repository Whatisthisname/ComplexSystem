import network_generation
import jax
import jax.numpy as jnp
import numpy as np
import ising_efficient
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # preparing comparable parameters
    N = 100
    # weight range changed from (-1,1.0) to (-0.1, 0.2) - individuals within the network tend to agree with their neighbours
    weight_range = (-0.04, 0.2)
    avg_deg = 20
    seed = 1
    n_steps = 500

    # Erdos_Renyi
    edge_prob = avg_deg/(N-1)

    # Scale free - Albert Barab'asi
    new_edges = avg_deg//2  # avg_deg ~ 2m

    # small world
    k_neighbours = avg_deg  # k neighbours per node
    rewire_prob = 0.1  # rewiring probability

    # Erdos-Renyi
    # adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
    # 	num_nodes=N, edge_prob=edge_prob, weight_range=weight_range
    # )

    # Scale-Free
    # adj_mat = network_generation.generate_scale_free_sparse_adjacency_matrix_jax(
    #     num_nodes=N, num_edge= new_edges, weight_range=weight_range
    # )

    # Small-World
    adj_mat = network_generation.generate_small_world_sparse_adjacency_matrix_jax(
        num_nodes=N, k=k_neighbours, p=rewire_prob, weight_range=weight_range
    )

    # Initialisation
    ones_init = jnp.negative(jnp.ones(shape=(len(adj_mat),)).astype(int))

    # External field
    alpha = 100  # time of the peak
    gamma = 5  # width of the peak

    def external_field(t, node_idx): return 10 * \
        ((node_idx[0] / len(adj_mat)) < l) * (jnp.exp(-((t-alpha) / gamma)**2))

    # lambda - fraction of people being exposed to the dynamic field
    l = 0.4  # % get the field

    µ = 0.9  # memory coefficient
    beta = 1.0  # temperature

    # cluster analysis parameters
    interval = 5
    trials = 100

    mags = []

    for l in ld_range
    epochs = 10
    for i in tqdm(range(epochs), position=0):
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            external_field=external_field,
            init_state=ones_init,  # np.copy(random_init), #or
            µ=µ,
            beta=1.1,
        )
        result = network.run_for_steps(n_steps, seed=i)

        # coeffs = approx_global_coefficients_over_time(result, nbs, trials=20, interval=interval)
        coeffs = np.mean(get_local_coefficients_over_time(
            result, nbs, interval=interval), axis=1)
        coeffs_over_time.append(coeffs)
        magnetization_erdos = np.mean(result, axis=1)
        mags.append(np.copy(magnetization_erdos))


if __name__ == "__main__":
    main()
