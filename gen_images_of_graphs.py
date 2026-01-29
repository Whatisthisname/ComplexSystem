import networkx as nx
import matplotlib.pyplot as plt
import network_generation

for topology in ["Erdos", "Small-World", "Scale-Free"]:
    nodes = 20
    edge_prob = 0.3
    expected_edges = int(edge_prob * (nodes * (nodes - 1)) / 2)
    print(expected_edges)
    if topology == "Erdos":
        adj_mat = network_generation.generate_erdos_renyi_sparse_adjacency_matrix(
            num_nodes=nodes, edge_prob=edge_prob, weight_range=(-0.04, 0.2)
        )
    elif topology == "Scale-Free":
        adj_mat = network_generation.generate_scale_free_sparse_adjacency_matrix_jax(
            num_nodes=nodes, num_edge=3, weight_range=(-0.04, 0.2)
        )
    elif topology == "Small-World":
        adj_mat = network_generation.generate_small_world_sparse_adjacency_matrix_jax(
            num_nodes=nodes,
            k=2 * int(expected_edges / nodes),
            p=0.1,
            weight_range=(-0.04, 0.2),
        )
    else:
        raise Exception("Invalid topology")

    # holds the ndarray of undirected adj matrix
    mat = adj_mat._adjacency_matrix

    # Convert adjacency matrix to NetworkX graph
    G = nx.from_numpy_array(mat)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw_forceatlas2(
        G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500
    )
    plt.title(f"{topology} Topology")
    plt.savefig(f"results/{topology}_graph.png")  # Save the graph as an image
