import sparse_adj_mat
import jax
import jax.numpy as jnp


def generate_erdos_renyi_sparse_adjacency_matrix(
    num_nodes: int, edge_prob: float, weight_range=(-1.0, 1.0), seed: int = 0
) -> sparse_adj_mat.Sparse_Adjacency_Matrix:
    """
    Generate an Erdos-Renyi random graph as a sparse adjacency matrix.

    :param num_nodes: Number of nodes in the graph.
    :type num_nodes: int
    :param edge_prob: Probability of an edge existing between any two nodes.
    :type edge_prob: float
    :param weight_range: Tuple specifying the range of edge weights (min, max). A value will be uniformly sampled from this.
    :param seed: Random seed for reproducibility.
    :type seed: int
    :return: A sparse adjacency matrix representing the generated graph.
    :rtype: sparse_adj_mat.Sparse_Adjacency_Matrix
    """
    mi, ma = weight_range
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)
    adj_mat = jax.random.uniform(
        key=key1, minval=mi, maxval=ma, shape=(num_nodes, num_nodes)
    )
    adj_mat *= jax.random.uniform(key=key2, shape=(num_nodes, num_nodes)) < edge_prob
    adj_mat *= jnp.triu(jnp.ones((num_nodes, num_nodes)), k=1)
    adj_mat += adj_mat.T

    return sparse_adj_mat.Sparse_Adjacency_Matrix(adj_mat)
