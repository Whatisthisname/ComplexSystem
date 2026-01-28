import sys
import os

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
import voronoi
import pickle
import os
from scipy.stats.qmc import Halton


def the_function(population_fraction: float, memory: float, steps: int) -> float:
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=100, edge_prob=20 / 100.0, weight_range=(-0.04, 0.2)
    )
    # weight range changed from (-1,1.0) to (-0.1, 0.2) - individuals within the network tend to agree with their neighbours

    ones_init = jnp.ones(shape=(len(adj_mat),)).astype(int)

    alpha = 100  # time of the peak
    gamma = 5  # width of the peak

    # population_fraction - fraction of people being exposed to the dynamic field

    signs = []
    for i in range(25):
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            external_field=lambda t, node_idx: 10
            * ((node_idx[0] / len(adj_mat)) < population_fraction)
            * (jnp.exp(-(((t - alpha) / gamma) ** 2))),
            init_state=-ones_init,
            µ=memory,
            beta=1.5,  # 1.1
            μ_is_weighted_according_to_neighborhood_size=False,
        )

        result = network.run_for_steps(steps, seed=i)
        magnetization_erdos = np.mean(result[-1])
        signs.append(np.sign(magnetization_erdos))
    print("fin")

    return np.mean(np.array(signs) > 0.0)


n = 20
low_res = [
    (p1, p2) for p1 in np.linspace(0.0, 1.0, n) for p2 in np.linspace(0.0, 1.0, n)
]

# xmin = 0.0 # fraction
# xmax = 0.3
# ymin = 0.6 # memory
# ymax = 1.0
xmin = 0.0  # fraction
xmax = 1.0
ymin = 0.0  # memory
ymax = 1.0

halton_sampler = Halton(d=2, scramble=False)
halton_seq = halton_sampler.random(n)
halton_seq = halton_seq * [(xmax - xmin), (ymax - ymin)] + [xmin, ymin]
all_pairs = halton_seq
all_pairs = [(x[0], x[1]) for x in all_pairs]

file_name = "memory_vs_fraction_temp_1.5_v2.pkl"

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
    colorbarlabel="Fraction of positive final magnetizations across runs",
    xlabel="Fraction of Population exposed to event",
    ylabel="Memory coefficient",
    title=r"Memory Experiment with $\beta = 1.5$",
)
