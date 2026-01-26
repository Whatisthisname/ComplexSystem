import numpy as np
import matplotlib.pyplot as plt


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

class ExternalField:
    def __init__(self, amplitude=1.0, omega=0.5, sigma=0.0, seed=None):
    
        self.amplitude = amplitude
        self.omega = omega
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, t, i=None):
        deterministic = self.amplitude * np.sin(self.omega * t)
        noise = self.sigma * self.rng.normal()
        return deterministic + noise

class BeliefState:
    def __init__(
        self,
        sparse_adj: Sparse_Adjacency_Matrix,
        field: np.ndarray | None,
        init_state=None,
        µ: float = 0.5,
        beta: float = 0.5,
        external_field=None,
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
        self.external_field = external_field
        
        self.magnetizations = [np.mean(self.state)]
        self.all_states = [self.init_state]

    def step(self):
        node_order = np.random.permutation(np.arange(len(self.graph_adjacency_mat)))
        ss = np.array([-1, 0, 1])

        for i in node_order:
            nbs, wght = self.graph_adjacency_mat.get_neighbors_and_weights(i)
            nbs_states = self.state[nbs]
            
            ext = 0.0
            if self.external_field is not None:
                ext = self.external_field(self.t, i)
            eff_i = wght @ nbs_states + self.field[i] + ext + self.µ * self.prev_state[i]
            
            sample_p = np.exp(self.beta * ss * eff_i)
            sample_p /= np.sum(sample_p)
            sample = np.random.choice(ss, p=sample_p)
    
            self.prev_state[i] = self.state[i]
            self.state[i] = sample
        self.all_states.append(np.copy(network))

        m = np.mean(self.state)
        self.magnetizations.append(m)
        self.t += 1

    def run(self, steps: int):
        for _ in range(steps):
            self.step()


adj = generate_erdos_renyi_adjacency_mat(num_nodes=200, edge_prob=0.1)
ext_field = ExternalField(
    amplitude=0.5,
    omega=0.2,    
    sigma=0.05,  
    seed=42
)
ext_field = lambda t, *i: 0.5 * np.sin(t) + np.random.rand()
network = BeliefState(
    sparse_adj=adj,
    field=np.zeros(len(adj)) - 1,
    init_state=np.zeros(len(adj)) + 1,
    µ=0,
    beta=0.5,
    external_field=ext_field,
)

network.run(steps=50)

# will hold, for each timestep, a list of numbers in [-1, 0, 1] with the corresponding state for that node at that time
network.all_states

# need graph algorithm to detect / extract connected components. This will involve the "get_neighbors_and_weights" function on the adjacency matrix object.


plt.plot(range(len(network.magnetizations)), network.magnetizations)
plt.title("Magnetization over time")
plt.show()

T = 100
h_vals = [ext_field(t) for t in range(T)]

plt.figure()
plt.plot(range(T), h_vals)
plt.title("External field h(t)")
plt.xlabel("time")
plt.ylabel("h(t)")
plt.show()

class BeliefState:
    def __init__(
        self,
        sparse_adj,
        field=None,
        init_state=None,
        mu: float = 0.5,
        beta: float = 0.5,
        external_field=None,
    ):
        self.sparse_adj = sparse_adj
        self.state = init_state.copy()
        self.mu = mu
        self.beta = beta
        self.external_field = external_field
        self.t = 0

        self.magnetizations = []
        self.record_magnetization()

    def record_magnetization(self):
        self.magnetizations.append(np.mean(self.state))

    def step(self):
        h_ext = self.external_field(self.t) if self.external_field else 0

        noise = np.random.normal(scale=1/self.beta, size=len(self.state))
        field = self.mu * self.state + h_ext + noise

        self.state = np.sign(field)
        self.state[self.state == 0] = 1

        self.t += 1
        self.record_magnetization()

def run_and_measure(h, mu, beta, steps=300):
    network = BeliefState(
        sparse_adj=adj,
        init_state=np.random.choice([-1, 0, 1], size=500),
        mu=mu,
        beta=beta,
        external_field=lambda t: h
    )
    for _ in range(steps):
        network.step()
    return np.mean(network.magnetizations[-10:])




h_vals = np.linspace(-1, 1, 25)
mu_vals = np.linspace(0, 1, 25)

M = np.zeros((len(mu_vals), len(h_vals)))

for i, mu in enumerate(mu_vals):
    for j, h in enumerate(h_vals):
        M[i, j] = run_and_measure(h=h, mu=mu, beta=5.0)

plt.figure(figsize=(6,5))
plt.imshow(
    M,
    origin="lower",
    aspect="auto",
    extent=[h_vals[0], h_vals[-1], mu_vals[0], mu_vals[-1]],
    cmap="coolwarm"
)
plt.colorbar(label="Magnetization m")
plt.xlabel("External field h")
plt.ylabel("Memory coefficient μ")
plt.title("Phase diagram of magnetization")
plt.show()

alpha = 250
gamma = 5

network = BeliefState(
    sparse_adj=adj,
    init_state=np.random.choice([-1, 0, 1], size=500),
    mu=0.8,
    beta=.5,
    # external_field=lambda t: 1.0 if alpha < t <= alpha + gamma  else 0.0
    external_field=lambda t: np.exp(-((t-alpha) / gamma)**2)
)

for _ in range(500):
    network.step()

ms = np.array(network.magnetizations)
t = np.arange(len(ms))

plt.figure(figsize=(7,3))
plt.step(t, (ms), where="post")
plt.ylim(-1.2, 1.2)
plt.xlabel("Time step")
plt.ylabel("sign(m)")
plt.title("Intermittent magnetization")
plt.show()



def generate_2D_plot(
    param1,
    param2,
    build_network_func,
    extract_metric,
    steps,
):
    grid = np.zeros((len(param1), len(param2)))

    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            net = build_network_func(p1, p2)
            net.run(steps)
            grid[i, j] = extract_metric(net)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")

    fig.colorbar(im, ax=ax, label="Final magnetization")
    ax.set_xticks(np.arange(len(param2)))
    ax.set_xticklabels(param2, rotation=45)
    ax.set_yticks(np.arange(len(param1)))
    ax.set_yticklabels(np.round(param1, 2))

    ax.set_xlabel("External field activation time $t_0$")
    ax.set_ylabel("Coupling strength $J$")
    ax.set_title("Effect of delayed external field on polarization")

    fig.tight_layout()

    return grid, fig