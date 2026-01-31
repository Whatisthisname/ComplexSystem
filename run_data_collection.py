
import jax
import jax.numpy as jnp
import numpy as np

import ising_efficient
import network_generation
from tqdm import tqdm
import pickle

import os


def main():
    # preparing comparable parameters
    N = 100
    # weight range changed from (-1,1.0) to (-0.1, 0.2) - individuals within the network tend to agree with their neighbours
    weight_range = (-0.04, 0.2)
    avg_deg = 20
    seed = 1
    n_steps = 250

    # Erdos_Renyi
    edge_prob = avg_deg / (N - 1)

    # Scale free - Albert Barab'asi
    new_edges = avg_deg // 2  # avg_deg ~ 2m

    # small world
    k_neighbours = avg_deg  # k neighbours per node
    rewire_prob = 0.1  # rewiring probability

    # Erdos-Renyi
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=N, edge_prob=edge_prob, weight_range=weight_range
    )

    # Scale-Free
    # adj_mat = network_generation.generate_scale_free_sparse_adjacency_matrix_jax(
    #     num_nodes=N, num_edge= new_edges, weight_range=weight_range
    # )

    # Small-World
    # adj_mat = network_generation.generate_small_world_sparse_adjacency_matrix_jax(
    #     num_nodes=N, k=k_neighbours, p=rewire_prob, weight_range=weight_range
    # )

    # Initialisation
    ones_init = jnp.negative(jnp.ones(shape=(len(adj_mat),)).astype(int))

    # External field
    alpha = 100  # time of the peak
    gamma = 5  # width of the peak

    def external_field(t, node_idx): return 10 * \
        ((node_idx[0] / len(adj_mat)) < l) * (jnp.exp(-((t - alpha) / gamma)**2))

    # lambda - fraction of people being exposed to the dynamic field
    l = 0.4  # % get the field

    n_frames = 100
    l_space = np.linspace(.0, .35, n_frames)

    µ = 0.9  # memory coefficient
    beta = 1.0  # temperature

    n_runs = 20

    # walk over parameter lambda
    for l in tqdm(l_space, position=0):
        data = []
        for seed in tqdm(range(n_runs), position=1):
            network = ising_efficient.BeliefNetwork(
                sparse_adj=adj_mat,
                external_field=external_field,
                init_state=ones_init,  # np.copy(random_init), #or
                µ=µ,
                beta=1.1,
            )
            data.append(network.run_for_steps(n_steps, seed=seed))

        filename = f"./results/lambda_walk/erdos_renyi_la_{np.round(l, 3)}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # create storage file
        with open(filename, "wb") as file:  # store in designated folder and file
            pickle.dump(np.asarray(data), file)


if __name__ == "__main__":
    main()
