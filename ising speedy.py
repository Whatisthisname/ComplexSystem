import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class Sparse_Adjacency_Matrix:
    def __init__(self, adjacency_matrix: np.ndarray):
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
        start = self.indptr[col_idx]
        end = self.indptr[col_idx + 1]
        return self.indices[start:end], self.data[start:end]

    def __len__(self):
        return self.n_rows


def generate_erdos_renyi_adjacency_mat(
    num_nodes, edge_prob, weight_range=(-1.0, 1.0)
) -> Sparse_Adjacency_Matrix:
    mi, ma = weight_range

    adj_mat = np.random.uniform(low=mi, high=ma, size=(num_nodes, num_nodes))
    adj_mat *= np.random.uniform(size=(num_nodes, num_nodes)) < edge_prob
    adj_mat *= np.triu(np.ones((num_nodes, num_nodes)), k=1)
    adj_mat += adj_mat.T

    print(adj_mat)

    return Sparse_Adjacency_Matrix(adj_mat)


class BeliefState:
    def __init__(
        self,
        sparse_adj: Sparse_Adjacency_Matrix,
        field: np.ndarray | None,
        init_state=None,
        µ: float = 0.5,
        beta: float = 0.5,
    ):
        if init_state is None:
            init_state = np.random.randint(-1, 2, size=len(sparse_adj))
        if field is None:
            field = np.zeros(len(sparse_adj))

        self.field = field
        self.init_state = init_state
        self.state = self.init_state
        self.prev_state = self.state
        self.graph_adjacency_mat = sparse_adj
        self.µ = µ
        self.beta = beta
        self.t = 0
        self.magnetizations = [np.mean(self.state)]
        self.all_states = [self.init_state]

    def step(self):
        node_order = np.random.permutation(np.arange(len(self.graph_adjacency_mat)))
        ss = np.array([-1, 0, 1])

        for i in node_order:
            nbs, wght = self.graph_adjacency_mat.get_neighbors_and_weights(i)
            nbs_states = self.state[nbs]
            eff_i = wght @ nbs_states + self.field[i] + self.µ * self.prev_state[i]

            sample_p = np.exp(self.beta * ss * eff_i)
            sample_p /= np.sum(sample_p)
            sample = np.random.choice(ss, p=sample_p)

            self.prev_state[i] = self.state[i]
            self.state[i] = sample
        self.all_states.append(np.copy(self.state))

        m = np.mean(self.state)
        self.magnetizations.append(m)
        self.t += 1

    def run(self, steps: int):
        for _ in range(steps):
            self.step()




def connected_components_by_belief(
    sparse_adj: Sparse_Adjacency_Matrix,
    belief_state: np.ndarray,
    target_belief: int
):
    """
    Identify connected components in the induced subgraph
    consisting only of nodes with a given belief state.
    """
    visited = set()
    components = []

    for node in range(len(sparse_adj)):
        if belief_state[node] != target_belief or node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        component = [node]

        while queue:
            u = queue.popleft()
            neighbors, _ = sparse_adj.get_neighbors_and_weights(u)

            for v in neighbors:
                if (
                    belief_state[v] == target_belief
                    and v not in visited
                ):
                    visited.add(v)
                    queue.append(v)
                    component.append(v)

        components.append(component)

    return components
def bfs_distances(sparse_adj: Sparse_Adjacency_Matrix, source: int, allowed_nodes: set):
    """
    Compute shortest-path distances from a source node
    restricted to an induced subgraph.
    """
    distances = {source: 0}
    queue = deque([source])

    while queue:
        u = queue.popleft()
        neighbors, _ = sparse_adj.get_neighbors_and_weights(u)

        for v in neighbors:
            if v in allowed_nodes and v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)

    return distances
def component_diameter(
    sparse_adj: Sparse_Adjacency_Matrix,
    component: list
):
    """
    Compute the exact diameter of a connected component
    using BFS from each node.
    """
    if len(component) <= 1:
        return 0

    component_set = set(component)
    max_distance = 0

    for node in component:
        distances = bfs_distances(
            sparse_adj, node, component_set
        )
        local_max = max(distances.values())
        max_distance = max(max_distance, local_max)

    return max_distance
def belief_state_diameters_over_time(sparse_adj: Sparse_Adjacency_Matrix, all_states: list[np.ndarray], belief_values=(-1, 0, 1)):
    """
    For each timestep and belief value, compute the
    maximum diameter among connected components.
    """
    results = {
        belief: [] for belief in belief_values
    }

    for state in all_states:
        for belief in belief_values:
            components = connected_components_by_belief(
                sparse_adj, state, belief
            )

            if not components:
                results[belief].append(0)
                continue

            diameters = [
                component_diameter(sparse_adj, comp)
                for comp in components
            ]
            results[belief].append(max(diameters))

    return results


adj = generate_erdos_renyi_adjacency_mat(num_nodes=500, edge_prob=0.1)
network = BeliefState(
    sparse_adj=adj,
    field=np.zeros(len(adj)) - 1,
    init_state=np.zeros(len(adj)) + 1,
    µ=1.0,
    beta=5,
)

network.run(steps=50)

# will hold, for each timestep, a list of numbers in [-1, 0, 1] with the corresponding state for that node at that time
network.all_states

# need graph algorithm to detect / extract connected components. This will involve the "get_neighbors_and_weights" function on the adjacency matrix object.

diameters = belief_state_diameters_over_time(
    sparse_adj=adj,
    all_states=network.all_states
)
plt.figure(figsize=(8, 5))
for belief, values in longest_paths.items():
    plt.plot(values, label=f"Belief {belief}")

plt.xlabel("Time step")
plt.ylabel("Longest shortest path")
plt.title("The longest diamater per belief state")
plt.legend()
plt.show()
