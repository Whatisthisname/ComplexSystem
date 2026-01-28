import sys
import os
import typing

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
import voronoi
import pickle
import os
from scipy.stats.qmc import Halton


def the_function(
    population_fraction: float,
    beta: float,
    steps: int,
    topology: typing.Literal["Erdos", "Small-World", "Scale-Free"],
) -> float:
    nodes = 100
    expected_edges = int(0.2 * (nodes * (nodes - 1)) / 2)
    print(expected_edges)
    if topology == "Erdos":
        adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
            num_nodes=nodes, edge_prob=0.2, weight_range=(-0.04, 0.2)
        )
    elif topology == "Scale-Free":
        adj_mat = network_generation.generate_scale_free_sparse_adjacency_matrix_jax(
            num_nodes=nodes, num_edge=10, weight_range=(-0.04, 0.2)
        )
    elif topology == "Small-World":
        # For n nodes and k initial edges, the expected amount of edges m = n * k / 2. Set m = expected_edges and solve for k
        # to get k = 2*m/n = 2*expected_edges / n
        adj_mat = network_generation.generate_small_world_sparse_adjacency_matrix_jax(
            num_nodes=nodes,
            k=2 * int(expected_edges / nodes),
            p=0.1,
            weight_range=(-0.04, 0.2),
        )
    else:
        raise Exception(
            """'topology' argument is invalid. Must be one of "Erdos", "Small-World", or "Scale-Free" """
        )

    ones_init = jnp.ones(shape=(len(adj_mat),)).astype(int)

    alpha = 100  # time of the peak
    gamma = 5  # width of the peak

    # population_fraction - fraction of people being exposed to the dynamic field
    # high beta - sweeping gamma and memory

    signs = []
    for i in range(10):
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            # external_field=lambda t, node_idx: jnp.sin(t * 0.1),
            # external_field=lambda t, node_idx: ((node_idx[0] / len(adj_mat)) < l) * jnp.sin(t * 0.1),
            external_field=lambda t, node_idx: 10
            * ((node_idx[0] / len(adj_mat)) < population_fraction)
            * (jnp.exp(-(((t - alpha) / gamma) ** 2))),
            init_state=-ones_init,
            µ=1.0,
            beta=beta,
            μ_is_weighted_according_to_neighborhood_size=False,
        )

        result = network.run_for_steps(steps, seed=i)
        magnetization_erdos = np.mean(result[-1])
        signs.append(np.sign(magnetization_erdos))

    return np.mean(np.array(signs) > 0.0)


topology: typing.Literal["Erdos", "Scale-Free", "Small-World"] = "Erdos"

n_points = 100

xmin = 0.0  # fraction
xmax = 1.0
ymin = 0.1  # inv_temp # 0.1
ymax = 3.0

halton_sampler = Halton(d=2, scramble=False)
halton_seq = halton_sampler.random(n_points)
halton_seq = halton_seq * [(xmax - xmin), (ymax - ymin)] + [xmin, ymin]
all_pairs = halton_seq
all_pairs = [(x[0], x[1]) for x in all_pairs]

file_name = f"opinion_change_threshold_{topology.lower()}.pkl"

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
outcomes = [
    the_function(p1, p2, steps=250, topology=topology) for (p1, p2) in uncomputed_pairs
]
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
    xlabel="Fraction of Population\nexposed to event",
    ylabel="Inverse Temperature",
)
