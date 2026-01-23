import numpy as np
import networkx as nx
import imageio
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt


def generate_random_graph(num_nodes, edge_prob, weight_range=(-1.0, 1.0)):
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
    mi, ma = weight_range
    for u, v in graph.edges():
        graph[u][v]["weight"] = np.random.uniform(low=mi, high=ma)
    return graph


def get_neighbors_and_weights(graph, node_idx):
    neighbors = np.array(list(graph.neighbors(node_idx)))
    weights = np.array([graph[node_idx][neighbor]["weight"] for neighbor in neighbors])
    return neighbors, weights


class BeliefState:
    def __init__(
        self,
        graph: nx.Graph,
        field: np.ndarray | None,
        init_state=None,
        µ: float = 0.5,
        beta: float = 0.5,
    ):
        if init_state is None:
            init_state = np.random.randint(-1, 2, size=len(graph))
        if field is None:
            field = np.zeros(len(graph))

        self.field = field
        self.init_state = init_state
        self.state = self.init_state
        self.prev_state = self.state
        self.graph = graph
        self.µ = µ
        self.beta = beta
        self.layout = nx.spring_layout(self.graph)
        self.t = 0
        self.magnetizations = [np.mean(self.state)]

    def step(self):
        node_order = np.random.permutation(np.arange(len(self.graph)))
        ss = np.array([-1, 0, 1])

        for i in node_order:
            nbs, wght = get_neighbors_and_weights(self.graph, i)
            nbs_states = self.state[nbs]
            eff_i = wght @ nbs_states + self.field[i] + self.µ * self.prev_state[i]

            sample_p = np.exp(self.beta * ss * eff_i)
            sample_p /= np.sum(sample_p)
            sample = np.random.choice(ss, p=sample_p)

            self.prev_state[i] = self.state[i]
            self.state[i] = sample

        m = np.mean(self.state)
        self.magnetizations.append(m)
        self.t += 1

    def visualize(self, ax=None):
        color_map = {1: "red", 0: "gray", -1: "blue"}  # Map states to colors
        node_colors = [color_map[self.state[node]] for node in self.graph.nodes()]

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 12))

        nx.draw(
            self.graph,
            self.layout,
            with_labels=True,
            node_color=node_colors,
            edge_color="black",
            node_size=500,
            font_size=10,
            ax=ax,
        )

        ax.set_title(f"Step t= {self.t}", fontsize=16)
        return ax

    def generate_gif(self, steps, filename="simulation.gif"):
        with TemporaryDirectory() as tempdir:
            filenames = []
            for step in range(steps):
                fig, ax = plt.subplots()
                self.visualize(ax=ax)
                filepath = f"{tempdir}/frame_{step}.png"
                plt.savefig(filepath)
                filenames.append(filepath)
                plt.close(fig)
                self.step()

            # Create GIF
            images = [imageio.imread(fname) for fname in filenames]
            imageio.mimsave(filename, images, duration=5)


g = generate_random_graph(num_nodes=5, edge_prob=1.0)
state = BeliefState(
    graph=g, field=np.zeros(len(g)) - 1, init_state=np.zeros(len(g)) + 1, µ=1.0, beta=5
)
state.generate_gif(steps=50)

plt.plot(range(len(state.magnetizations)), state.magnetizations)
plt.title("Magnetization over time")
plt.show()


# state = [-1, -1, -1, -1, -1]
# adj = [0, 1, 0, 0, 0]
#       [1, 0, 1, 0, 0]
#       [0, 1, 0, 1, 0]
#       [0, 0, 1, 0, 1]
#       [0, 0, 0, 1, 0]
