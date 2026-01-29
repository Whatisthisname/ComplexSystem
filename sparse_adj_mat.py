import numpy as np
import jax.numpy as jnp


class Sparse_Adjacency_Matrix:
    """A compact storage format for adjacency matrices that allows efficient retrieval of neighbors from a given node."""

    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Construct an instance of the class. Takes an adjacency matrix and packs it compactly in memory, allowing fast traversal of local neighborhoods.

        :param adjacency_matrix: Some symmetric square matrix with floating point entries corresponding to edge weights. Self loops not allowed.
        :type adjacency_matrix: np.ndarray
        """
        n = adjacency_matrix.shape[0]
        assert np.allclose(
            (adjacency_matrix + adjacency_matrix.T) * 0.5, adjacency_matrix
        ), "The passed adjacency matrix should be symmetric but it seems not to be."
        assert np.allclose(np.zeros(n), np.diag(adjacency_matrix)), (
            "The adjacency matrix should be all zeros on the diagonal, meaning no self-loops in the graph, but there are some nonzeros."
        )
        adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.T)
        self.n_rows, self.n_cols = adjacency_matrix.shape
        self.data = []
        self.indices = []
        self.indptr = [0]

        for col in range(self.n_cols):
            col_data = []
            col_indices = []
            for row in range(self.n_rows):
                value = adjacency_matrix[row, col]
                if value != 0:
                    col_data.append(value)
                    col_indices.append(row)
            self.data.extend(col_data)
            self.indices.extend(col_indices)
            self.indptr.append(len(self.data))

    def get_neighbors_and_weights(self, col_idx):
        """
        Retrieves the neighbors and their corresponding weights for a given node.

        :param col_idx: The index of the node (column) for which neighbors and weights are retrieved.
        :return: A tuple of two lists:

                - `indices`: List of neighbor indices.

                - `data`: List of weights corresponding to the edges to the neighbors.
        """
        start = self.indptr[col_idx]
        end = self.indptr[col_idx + 1]
        return self.indices[start:end], self.data[start:end]

    def precompute_neighbors_and_weights(self):
        """
        Precomputes fixed-size arrays for neighbors and weights of each node. This is useful for efficient computation and memory layout in the simulation.

        This method calculates two 2D arrays:
        - `neighbors`: Contains the indices of neighbors for each node, padded with -1 for missing neighbors.
        - `weights`: Contains the weights of edges to neighbors, padded with 0 for missing edges.

        The maximum degree of the graph is inferred to determine the second dimension "m" of the returned arrays, which are both (n, m) with "n" the amount of nodes.

        :return: A tuple of two JAX arrays (`neighbors`, `weights`).
        """
        max_degree = max(
            self.indptr[i + 1] - self.indptr[i] for i in range(self.n_cols)
        )

        neighbors = -np.ones((self.n_cols, max_degree), dtype=int)
        weights = np.zeros((self.n_cols, max_degree), dtype=float)

        for col_idx in range(self.n_cols):
            start = self.indptr[col_idx]
            end = self.indptr[col_idx + 1]
            degree = end - start

            neighbors[col_idx, :degree] = self.indices[start:end]
            weights[col_idx, :degree] = self.data[start:end]

        return jnp.asarray(neighbors), jnp.asarray(weights)

    def __len__(self):
        return self.n_rows
