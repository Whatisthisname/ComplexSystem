
import network_generation
import jax
import jax.numpy as jnp
import numpy as np
import ising_efficient
from clustering_coefficient import approx_global_coefficients_over_time, get_local_coefficients_over_time
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # preparing comparable parameters
    N = 100
    weight_range = (-0.04, 0.2)  # weight range changed from (-1,1.0) to (-0.1, 0.2) - individuals within the network tend to agree with their neighbours 
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
    def external_field(t, node_idx): return 10 * ((node_idx[0] / len(adj_mat)) < l) * (jnp.exp(-((t - alpha) / gamma)**2))

    # lambda - fraction of people being exposed to the dynamic field
    beta = 1.5  # temperature
    l = 0.6  # % get the field
    plotname = "vulnerable"

    µ = 1  # memory coefficient

    interval = 2
    trials = 100
    epochs = 10

    x_ticks = list(range(0, n_steps + 1, interval))
    nbs, _ = adj_mat.precompute_neighbors_and_weights()
    nbs = [np.asarray(nb[nb > -1]) for nb in nbs]
    coeffs_over_time = []
    mags = []
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 6))
    for i in tqdm(range(epochs), position=0):
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            #external_field=lambda t, node_idx: jnp.sin(t * 0.1),
            #external_field=lambda t, node_idx: ((node_idx[0] / len(adj_mat)) < l) * jnp.sin(t * 0.1),
            external_field=external_field,
            init_state=ones_init,  # np.copy(random_init), #or 
            µ=µ,
            beta=beta,
        )
        result = network.run_for_steps(n_steps, seed=i)

        # coeffs = approx_global_coefficients_over_time(result, nbs, trials=trials, interval=interval)
        coeffs = np.mean(get_local_coefficients_over_time(result, nbs, interval=interval), axis=1)
        coeffs_over_time.append(coeffs)
        magnetization_erdos = np.mean(result, axis=1)
        mags.append(np.copy(magnetization_erdos))

    mags_array = np.array(mags)
    for mag in mags:
        axs[0].plot(mag, alpha=.4)
    axs[0].plot(np.mean(mags, axis=0), alpha=1, label="mean")
    axs[0].set_ylim(-1, 1)
    axs[0].set_ylabel("Magnetization")
    axs[0].set_title(f"Magnetization in {plotname} point:\nβ = {beta}, λ = {l}")
    axs[0].legend()

    for c in coeffs_over_time: 
        axs[1].plot(x_ticks, c, alpha=.3)
    axs[1].plot(x_ticks, np.mean(coeffs_over_time, axis=0), alpha=1, label="mean")
    axs[1].set_title("Average clustering coefficient\nTrials = 100, interval = 2")
    # axs[1].xlabel("Time steps")
    axs[1].set_ylabel("Clustering Coefficient")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.suptitle("Erdos-Renyi topology magnetisation over clustering")
    fig.supxlabel("Time steps")

    plt.savefig(f'results/erdos_clust_coeff_{plotname}_smaller.png')
    plt.show()


if __name__ == "__main__":
    main()
