
import jax
import jax.numpy as jnp
import numpy as np

import ising_efficient
import network_generation
import pickle
import os

from tqdm import tqdm


def main():
    # preparing comparable parameters
    n = 100
    # weight range changed from (-1,1.0) to (-0.1, 0.2) - individuals within the network tend to agree with their neighbours
    weight_range = (-0.04, 0.2)
    avg_deg = 20
    seed = 1
    n_steps = 250

    # erdos_renyi
    edge_prob = avg_deg / (n - 1)

    # scale free - albert barab'asi
    new_edges = avg_deg // 2  # avg_deg ~ 2m

    # small world
    k_neighbours = avg_deg  # k neighbours per node
    rewire_prob = 0.1  # rewiring probability

    # erdos-renyi
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=n, edge_prob=edge_prob, weight_range=weight_range
    )

    # initialisation
    ones_init = jnp.negative(jnp.ones(shape=(len(adj_mat),)).astype(int))

    # external field
    alpha = 100  # time of the peak
    gamma = 5  # width of the peak

    def external_field(t, node_idx): return 10 * \
        ((node_idx[0] / len(adj_mat)) < l) * (jnp.exp(-((t - alpha) / gamma)**2))

    n_frames = 100
    # lambda - fraction of people being exposed to the dynamic field
    l_space = np.linspace(.0, .35, n_frames)

    µ = 0.9  # memory coefficient
    beta = 1.0  # temperature

    n_runs = 20

    # walk over parameter lambda
    for l in tqdm(l_space, position=0):
        data = []
        for seed in tqdm(range(n_runs), position=1):
            network = ising_efficient.beliefnetwork(
                sparse_adj=adj_mat,
                external_field=external_field,
                init_state=ones_init,  # np.copy(random_init), #or
                µ=µ,
                beta=1.1,
            )
            data.append(network.run_for_steps(n_steps, seed=seed))

        filename = f"./results/lambda_walk/erdos_renyi_la_{np.round(l, 3)}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=true)  # create storage file
        with open(filename, "wb") as file:  # store in designated folder and file
            pickle.dump(np.asarray(data), file)


if __name__ == "__main__":
    main()
