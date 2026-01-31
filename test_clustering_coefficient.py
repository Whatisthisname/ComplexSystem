import pytest
import jax.numpy as jnp

from sparse_adj_mat import Sparse_Adjacency_Matrix
from clustering_coefficient import get_local_coefficients, approx_global_coefficient


def test_coefficient_fully_connected():
    n = 10
    adj = jnp.ones((n, n)) - jnp.eye(n)
    nbs, _ = Sparse_Adjacency_Matrix(adj).precompute_neighbors_and_weights()
    state = jnp.ones(shape=(n))

    assert jnp.allclose(get_local_coefficients(state, nbs), 1)
    assert jnp.isclose(approx_global_coefficient(state, nbs), 1)


def test_coefficient_fully_disconnected():
    n = 10
    adj = jnp.zeros(shape=(n, n))
    nbs, _ = Sparse_Adjacency_Matrix(adj).precompute_neighbors_and_weights()
    state = jnp.ones(shape=(n))

    assert jnp.allclose(get_local_coefficients(state, nbs), 0)
    assert jnp.isclose(approx_global_coefficient(state, nbs), 0)


def test_local_coefficient():
    adj = jnp.array([
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ])
    nbs, _ = Sparse_Adjacency_Matrix(adj).precompute_neighbors_and_weights()
    state = jnp.ones(shape=(len(adj)))

    answer = jnp.asarray([0, 1, 1 / 3, 1])

    assert jnp.allclose(get_local_coefficients(state, nbs), answer)


def test_global_coefficient():
    n_samples = 200
    adj = jnp.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ])
    nbs, _ = Sparse_Adjacency_Matrix(adj).precompute_neighbors_and_weights()
    state = jnp.ones(shape=(len(adj),))

    assert jnp.isclose(approx_global_coefficient(state, nbs, n_samples=n_samples), .75, rtol=1e-01)
