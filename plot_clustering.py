
import network_generation
import jax
import jax.numpy as jnp
import numpy as np
import ising_efficient
from clustering_coefficient import approx_global_coefficients_over_time, get_local_coefficients_over_time
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_clustering_plot(
    plotname: str, 
    beta: float, 
    l: float, 
    n_samples: int = 100, 
    interval: int = 5, 
    n_runs: int = 10
):
    """
    run n_runs and return figure of average magnetization and clustering coefficient
    """
    assert n_samples > 50
    assert interval > 0
    assert n_runs > 0

    # preparing comparable parameters
    N = 100
    avg_deg = 20
    n_steps = 250

    # Erdos_Renyi
    edge_prob = avg_deg / (N - 1) 
    adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
        num_nodes=N, edge_prob=edge_prob, weight_range=(-.04, .2)
    )

    # Initialisation
    ones_init = jnp.negative(jnp.ones(shape=(len(adj_mat),)).astype(int))

    # External field
    alpha = 100  # time of the peak
    gamma = 5  # width of the peak
    def external_field(t, node_idx): return 10 * ((node_idx[0] / len(adj_mat)) < l) * (jnp.exp(-((t - alpha) / gamma)**2))

    nbs, _ = adj_mat.precompute_neighbors_and_weights()
    coeffs_over_time = []
    mags = []
    for i in tqdm(range(n_runs), position=0):
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            external_field=external_field,
            init_state=ones_init,
            µ=1,
            beta=beta,
        )
        result = network.run_for_steps(n_steps, seed=i)

        coeffs = approx_global_coefficients_over_time(result, nbs, n_samples=n_samples, interval=interval)
        coeffs_over_time.append(coeffs)
        magnetization_erdos = np.mean(result, axis=1)
        mags.append(np.copy(magnetization_erdos))

    mags_array = np.array(mags)

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 6))
    # plot average magnetization
    for mag in mags:
        ax0.plot(mag, alpha=.4)
    ax0.plot(np.mean(mags, axis=0), alpha=1, label="mean")
    ax0.set_ylim(-1, 1)
    ax0.set_ylabel("Magnetization")
    ax0.set_title(f"Magnetization in {plotname} point:\nβ = {beta}, λ = {l}")
    ax0.legend()

    # plot approximated global clustering coefficients
    x_ticks = list(range(0, n_steps + 1, interval))
    for c in coeffs_over_time: 
        ax1.plot(x_ticks, c, alpha=.3)
    ax1.plot(x_ticks, np.mean(coeffs_over_time, axis=0), alpha=1, label="mean")
    ax1.set_title(f"Average clustering coefficient\nsamples = {n_samples}, interval = {interval}")
    ax1.set_ylabel("Clustering Coefficient")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.suptitle("Erdos-Renyi topology magnetization over clustering")
    fig.supxlabel("Time steps")
    return fig


def main():
    # example usage
    plotname = "recovery"
    beta = 1.5
    l = .1
    fig = generate_clustering_plot(plotname, beta, l, n_samples = 1000, interval = 10, n_runs=1)
    plt.show()


if __name__ == "__main__":
    main()
