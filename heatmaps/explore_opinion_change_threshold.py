import sys
import os

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
import ising_efficient
import network_generation
import heatmap



def the_function(population_fraction : float, beta : float, steps : int) -> float:
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=100, edge_prob=20/100.0, weight_range=(-0.1, 0.2)
    )
    # weight range changed from (-1,1.0) to (-0.1, 0.2) - individuals within the network tend to agree with their neighbours 

    ones_init = jnp.ones(shape=(len(adj_mat),)).astype(int)

    alpha = 25 #time of the peak
    gamma = 5 #width of the peak

    #population_fraction - fraction of people being exposed to the dynamic field
    #high beta - sweeping gamma and memory

    signs = []
    for i in range(10):

        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            #external_field=lambda t, node_idx: jnp.sin(t * 0.1),

            #external_field=lambda t, node_idx: ((node_idx[0] / len(adj_mat)) < l) * jnp.sin(t * 0.1),
            external_field=lambda t, node_idx: -10 * ((node_idx[0] / len(adj_mat)) < population_fraction) * (jnp.exp(-((t-alpha) / gamma)**2)),
            init_state=ones_init,
            µ=1.0,
            beta=beta,
        )

        result = network.run_for_steps(steps, seed= i)
        magnetization_erdos = np.mean(result[-1])
        signs.append(np.sign(magnetization_erdos))

    return np.mean(np.array(signs) > 0.0)

n = 10
fractions = np.linspace(0.0, 0.5, n)
betas = np.linspace(0.2, 1.5, n)

grid, fig = heatmap.generate_2D_plot(fractions, betas, steps=50, build_and_run_network=the_function)

plt.xlabel("Tempurerature beta")
plt.ylabel("Population fraction exposed to dynamic field")
plt.show()