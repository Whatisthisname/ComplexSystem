# Phenomenon to study:
Formation of bubbles and echo chambers in social networks.

# Model to study:

The most similar model is the Ising model.

We have a network where each node $n_i$ is associated with a scalar $s_i \in [-1, 1]$, called their "view".

The network is not necessarily a 2D regular grid, it will be interesting to study different sorts like the small-world graphs (Watts-Strogatz model possibly).

There is a local energy $$H_i := \sum_{n_j\in N(n_i)} -s_j  s_i$$

which, in expectation, by definition of suitable transition rule (TBD), will be decreasing or constant for any fixed node throughout the evolution of the system.

With a mechanism in the transition rule that modifies the network by adding and removing edges, it can simulate polarization and formation of bubbles.

If a view $s$ goes from $s$ to $s'$, then there could be a probability $p \propto \frac{1}{2}|s-s'|$ of edges aligned within the local cluster that are similar to view $s$ to get removed, while new edges might be created to nodes that are more closely related with the new view $s'$.

Add parameters for rewiring probability and initial graph,
measure clustering coefficient.

# Terms:

Innovation spread



