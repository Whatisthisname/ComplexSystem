import typing
from ising_efficient import BeliefNetwork
import numpy as np
import matplotlib.pyplot as plt
import network_generation

Run_And_Extract_Function = typing.Callable[[float, float, int], float]

def generate_2D_plot(
    param1 : float,              
    param2 : float,              
    steps : int,
    build_and_run_network : Run_And_Extract_Function,
    param1_name: str = "Param1",
    param2_name: str = "Param2",
):
    assert steps > 0
    grid = np.zeros((len(param1), len(param2)))

    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            progress = ((i) * len(param2) + (j + 1)) / (len(param1) * len(param2))
            print(f"Progress: {progress * 100}%")
            grid[i, j] = build_and_run_network(p1, p2, steps)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")

    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(param2)))
    ax.set_xticklabels(param2, rotation=45)
    ax.set_yticks(np.arange(len(param1)))
    ax.set_yticklabels(np.round(param1, 2))

    ax.set_ylabel(param1_name)
    ax.set_xlabel(param2_name)

    fig.tight_layout()
    return grid, fig

# Example usage
if __name__ == "__main__":
    
    def build_network_func(J, t0, steps : int):
    
        beta = 1.0
        def external_field(t, *node_idx):
            return 10 * (t >= t0) * J

        adj = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
            num_nodes=100, edge_prob=0.06
        )

        net = BeliefNetwork(
            sparse_adj=adj,
            init_state=np.zeros(len(adj), dtype=int),
            μ=0.5,
            beta=beta,
            external_field=external_field,
        )

        result = net.run_for_steps(steps, seed= 30)

        return np.mean(result[-1])
        
    n = 10

    J_values = np.linspace(0.1, 1.5, n)
    t0_values = np.linspace(0, 100, n, dtype=int)


    grid, fig = generate_2D_plot(
        param1=J_values,
        param2=t0_values,
        build_and_run_network=build_network_func,
        steps=100,
    )
    plt.show()