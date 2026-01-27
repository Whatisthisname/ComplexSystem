import sys
import os

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)
import ising_efficient
import network_generation
import heatmap
import voronoi
import pickle
import os


def the_function(population_fraction: float, memory: float, steps: int) -> float:
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=100, edge_prob=20 / 100.0, weight_range=(-0.1, 0.2)
    )
    # weight range changed from (-1,1.0) to (-0.1, 0.2) - individuals within the network tend to agree with their neighbours

    ones_init = jnp.ones(shape=(len(adj_mat),)).astype(int)

    alpha = 100  # time of the peak
    gamma = 5  # width of the peak

    # population_fraction - fraction of people being exposed to the dynamic field

    signs = []
    for i in range(10):
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            external_field=lambda t, node_idx: -10
            * ((node_idx[0] / len(adj_mat)) < population_fraction)
            * (jnp.exp(-(((t - alpha) / gamma) ** 2))),
            init_state=ones_init,
            µ=memory,
            beta=1.1,
            μ_is_weighted_according_to_neighborhood_size=False,
        )

        result = network.run_for_steps(steps, seed=i)
        magnetization_erdos = np.mean(result[-1])
        signs.append(np.sign(magnetization_erdos))

    return np.mean(np.array(signs) > 0.0)


n = 16
low_res = [
    (p1, p2) for p1 in np.linspace(0.0, 1.0, n) for p2 in np.linspace(0.0, 1.0, n)
]
# high_res = [
#     (p1, p2) for p1 in np.linspace(0.3, 1.0, 12) for p2 in np.linspace(0.5, 1.0, 5)
# ]
all_pairs = low_res

file_name = "memory_vs_fraction.pkl"

data_file = file_name
all_data = []

if os.path.exists(data_file):
    with open(data_file, "rb") as f:
        all_data = pickle.load(f)
    computed_pairs = set((x[0], x[1]) for x in all_data)
else:
    computed_pairs = set()
    all_data = []

# Filter out pairs that have already been computed
uncomputed_pairs = [pair for pair in all_pairs if pair not in computed_pairs]

# Compute outcomes for uncomputed pairs
outcomes = [the_function(p1, p2, steps=250) for (p1, p2) in uncomputed_pairs]
new_data = [(p1, p2, outcome) for (p1, p2), outcome in zip(uncomputed_pairs, outcomes)]

# Merge old and new data
all_data.extend(new_data)

with open(file_name, "wb") as f:
    pickle.dump(all_data, f)

pairs = np.array([[x[0], x[1]] for x in all_data])
fraction = np.array([x[2] for x in all_data])


voronoi.plot_triples(
    pairs,
    fraction,
    colorbarlabel="Fraction of Positive Final States",
    xlabel="Population Fraction Exposed",
    ylabel="Memory_coefficient",
)
