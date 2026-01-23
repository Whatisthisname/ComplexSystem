import ising_efficient
import network_generation
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
    num_nodes=10, edge_prob=0.1
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


result = network.run_for_steps(200)
magnetization = np.mean(result, axis=1)
plt.plot(magnetization)
plt.show()
