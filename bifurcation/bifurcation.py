import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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


def get_final_magnetization(inv_temp: float, seed: int) -> float:
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=100, edge_prob=0.2, weight_range=(-0.04, 0.2)
    )

    random_init = jax.random.randint(
        shape=(len(adj_mat),),
        minval=-1,
        maxval=2,
        key=jax.random.PRNGKey(np.random.randint(low=0, high=100)),
    )

    network = ising_efficient.BeliefNetwork(
        sparse_adj=adj_mat,
        external_field=lambda t, node_idx: 0.0,
        init_state=random_init,
        µ=0.0,
        beta=inv_temp,
        μ_is_weighted_according_to_neighborhood_size=True,
    )

    result = network.run_for_steps(100, seed=seed)
    magnetization = np.mean(result, axis=1)
    return np.mean(magnetization[-5:])


if False:
    pass
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=100, edge_prob=0.2, weight_range=(-0.04, 0.2)
    )

    random_init = jax.random.randint(
        shape=(len(adj_mat),),
        minval=-1,
        maxval=2,
        key=jax.random.PRNGKey(np.random.randint(low=0, high=100)),
    )

    network = ising_efficient.BeliefNetwork(
        sparse_adj=adj_mat,
        external_field=lambda t, node_idx: 0.0,
        init_state=random_init,
        µ=0.0,
        beta=4,
        μ_is_weighted_according_to_neighborhood_size=True,
    )

    result = network.run_for_steps(100, seed=5)
    magnetization = np.mean(result, axis=1)

    plt.plot(magnetization)
    plt.ylim(-1, 1)
    plt.show()

# exit()
inv_temps = np.linspace(0.1, 2.0, 15)
runs = 10
data = np.zeros((len(inv_temps), runs))
for i_temp, inv_temp in enumerate(inv_temps):
    for i_run in range(runs):
        data[i_temp, i_run] = get_final_magnetization(
            inv_temp, seed=np.random.randint(0, 100000)
        )

for run in range(runs):
    plt.scatter(inv_temps, data[:, run], color="black")

plt.ylim(-1, 1)
plt.xlabel("Inverse temperature")
plt.ylabel("Final average magnetization")
plt.show()
