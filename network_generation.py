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


#this code was developed with the use of GPT4 after prompting it with pseudo code
def generate_scale_free_sparse_adjacency_matrix_jax(
    num_nodes: int,
    num_edge: int,
    weight_range=(-1.0, 1.0),
    seed: int = 0,
):
    """
    Generate a scale-free network adjacency matrix using the
    Barabási–Albert preferential attachment model in JAX.

    num_nodes : int
        Total number of nodes in the network.

    num_edge : int
        Number of edges each newly added node attaches to existing nodes.
        Must satisfy num_edge >= 1.

    weight_range : (float, float)
        Range for symmetric edge weights.

    seed : int
        PRNG seed for reproducibility.

    Returns
    -------
    Sparse_Adjacency_Matrix
        Weighted symmetric adjacency matrix wrapped in a sparse-like class.
    """

    assert num_edge >= 1, "num_edge must be >= 1"
    assert num_nodes > num_edge + 1, "num_nodes must exceed initial core size"

    key = jax.random.PRNGKey(seed)
    w_min, w_max = weight_range

    # initial fully connected core size
    init_size = num_edge + 1

    # initialize adjacency matrix
    adj = jnp.zeros((num_nodes, num_nodes), dtype=jnp.float32)

    # initialize degrees
    degrees = jnp.zeros(num_nodes, dtype=jnp.float32)

    # create a fully connected initial core
    core_mask = jnp.triu(jnp.ones((init_size, init_size)), k=1)
    core_weights = jax.random.uniform(
        key, shape=(init_size, init_size), minval=w_min, maxval=w_max
    )

    core_adj = core_mask * core_weights
    core_adj = core_adj + core_adj.T

    adj = adj.at[:init_size, :init_size].set(core_adj)

    # update degree vector for initial core
    degrees = degrees.at[:init_size].set(jnp.sum(core_adj != 0, axis=1))

    # Node attachment function for lax.scan

    def attach_node(state, node_id):
        adj, degrees, key = state

        key, subkey_nodes, subkey_weights = jax.random.split(key, 3)

        # preferential attachment probabilities - create mask for available nodes
        available_mask = jnp.arange(num_nodes) < node_id
        degree_slice = jnp.where(available_mask, degrees, 0)
        probs = degree_slice / (jnp.sum(degree_slice) + 1e-10)  # avoid division by zero
        
        # sample attachment targets from all possible targets (with masking)
        # Use gumbel-max trick to sample without replacement in a differentiable way
        gumbels = -jnp.log(-jnp.log(jax.random.uniform(subkey_nodes, shape=(num_nodes,))))
        # mask out unavailable nodes and nodes we've already selected
        masked_gumbels = jnp.where(available_mask, gumbels + jnp.log(probs + 1e-10), -jnp.inf)
        # get top num_edge indices
        targets = jnp.argsort(-masked_gumbels)[:num_edge]

        # sample edge weights
        weights = jax.random.uniform(
            subkey_weights, shape=(num_edge,), minval=w_min, maxval=w_max
        )

        # update adjacency matrix symmetrically
        adj = adj.at[node_id, targets].set(weights)
        adj = adj.at[targets, node_id].set(weights)

        # update degree counts
        degrees = degrees.at[node_id].set(num_edge)
        degrees = degrees.at[targets].add(1)

        return (adj, degrees, key), None

    # sequentially grow the network
    (adj, degrees, _), _ = jax.lax.scan(
        attach_node,
        (adj, degrees, key),
        jnp.arange(init_size, num_nodes),
    )
    return sparse_adj_mat.Sparse_Adjacency_Matrix(adj)


#this code was developed with the use of GPT4 after prompting it with pseudo code
def generate_small_world_sparse_adjacency_matrix_jax(
    num_nodes: int,
    k: int,
    p: float,
    weight_range= (-1.0, 1.0),
    seed: int = 0,
):
    """
    Generate a small-world network using the Watts–Strogatz model,
    implemented in JAX.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the network.
    k : int
        Number of nearest neighbors per node (must be even).
    p : float
        Rewiring probability.
    weight_range : (float, float)
        Included for interface consistency (unused; edges are unweighted).
    seed : int
        PRNG seed for reproducibility.

    Returns
    -------
    Sparse_Adjacency_Matrix
        Symmetric unweighted adjacency matrix.
    """
    #checks
    assert k % 2 == 0, "k must be even for ring lattice construction"
    assert 0.0 <= p <= 1.0, "p must lie in [0, 1]"
    assert k < num_nodes, "k must be smaller than num_nodes"

    key = jax.random.PRNGKey(seed)

    # 1. Initialize empty adjacency matrix
    adj = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int8)

    # 2. Construct regular ring lattice
    # Each node connects to k/2 neighbors on each side

    def connect_ring(i, adj):
        neighbors = (i + jnp.arange(1, k // 2 + 1)) % num_nodes
        adj = adj.at[i, neighbors].set(1)
        adj = adj.at[neighbors, i].set(1)
        return adj

    adj = jax.lax.fori_loop(0, num_nodes, connect_ring, adj)

    # 3. Rewiring(Watts–Strogatz)
    # Only forward edges are considered to avoid double rewiring

    def rewire_edge(state, idx):
        adj, key = state
        i, j = idx

        key, subkey_rewire, subkey_target = jax.random.split(key, 3)

        # decide whether to rewire
        do_rewire = jax.random.bernoulli(subkey_rewire, p)

        def rewire(adj):
            # remove existing edge
            adj = adj.at[i, j].set(0)
            adj = adj.at[j, i].set(0)

            # candidate nodes excluding self and existing neighbors
            invalid = adj[i] + jnp.eye(num_nodes)[i]
            probs = (1.0 - invalid) / jnp.sum(1.0 - invalid)

            new_j = jax.random.choice(subkey_target, num_nodes, p=probs)

            adj = adj.at[i, new_j].set(1)
            adj = adj.at[new_j, i].set(1)

            return adj

        adj = jax.lax.cond(do_rewire, rewire, lambda x: x, adj)
        return (adj, key), None

    # generate list of edges to consider (i < j only)
    edges_i = jnp.repeat(jnp.arange(num_nodes), k // 2)
    edges_j = (edges_i + jnp.tile(jnp.arange(1, k // 2 + 1), num_nodes)) % num_nodes
    edge_indices = jnp.stack([edges_i, edges_j], axis=1)

    (adj, _), _ = jax.lax.scan(
        rewire_edge,
        (adj, key),
        edge_indices,
    )

    #weigth range
    w_min, w_max = weight_range
    #assign random symmetric weigths to existing edges
    key1, _ = jax.random.split(key)
    weights = jax.random.uniform(
        key=key1, minval=w_min, maxval=w_max, shape=(num_nodes, num_nodes)
    )
    weighted_adj = adj * weights
    # Make symmetric by averaging
    weighted_adj = 0.5 * (weighted_adj + weighted_adj.T)

    return sparse_adj_mat.Sparse_Adjacency_Matrix(weighted_adj)