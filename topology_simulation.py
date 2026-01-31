
import matplotlib.pyplot as plt
import jax.numpy as jnp
import ising_efficient
from ising_efficient import BeliefNetwork
import numpy as np


def simulation_topology(adj_mat, topology_name, param_name, beta, l):
    #initialisation 
    ones_init = jnp.negative(jnp.ones(shape=(len(adj_mat),)).astype(int)) #intitialisation with -1

    #external_field
    external_field = lambda t, node_idx: 10 * ((node_idx[0] / len(adj_mat)) < l) * (jnp.exp(-((t-alpha) / gamma)**2))

    mags = []
    for i in range(10):
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            external_field=external_field,
            init_state=ones_init, 
            µ=µ,
            beta=beta,
        )
        result = network.run_for_steps(250, seed=i)

        magnetization_erdos = np.mean(result, axis=1)
        mags.append(np.copy(magnetization_erdos))
    mags_array = np.array(mags)

    #plot 
    plt.figure(figsize=(15,9))
    for m in mags_array: 
        plt.plot(m)

    plt.xlabel("Time steps")
    plt.ylabel("Magnetization")
    plt.ylim(-1,1)
    plt.title(f"Impact of {topology_name} topology on belief alignment \n {param_name} point: β = {beta}, λ = {l}")
    plt.grid(True, alpha=0.3)
    filename = f"results/{topology_name.replace('-', '_')}_{param_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.show()


def simulation_no_external_field_stacked(adj_mat, topology_name, param_name, beta, l=0):
    #initialisation 
    results_list = []
    random_init = jax.random.randint(shape=(len(adj_mat),),minval=-1,maxval=2,key=jax.random.PRNGKey(np.random.randint(low=0, high=100000)),)
    
    network = ising_efficient.BeliefNetwork(
        sparse_adj=adj_mat,
        external_field=lambda t, node_idx: 0,
        init_state=np.copy(random_init),
        µ=µ,
        beta=beta,
    )
    result = network.run_for_steps(250, seed=0)
    results_list.append(np.array(result))

    magnetization_erdos = np.mean(result, axis=1)
    
    #results for the stackchart
    first_result = results_list[0]  # shape: (timesteps, n_nodes)
    time_steps = np.arange(first_result.shape[0])
    counts_neg = np.sum(first_result == -1, axis=1)
    counts_zero = np.sum(first_result == 0, axis=1)
    counts_pos = np.sum(first_result == 1, axis=1)

    #plot
    fig, ax1 = plt.subplots(figsize=(15,9))
     #trajectory
    ax1.plot(magnetization_erdos, color = 'black', linewidth =2)
    ax1.set_ylabel("Magnetization")
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("Time steps")
    #stacked bacground counts
    ax2 = ax1.twinx()
    ax2.stackplot(time_steps, counts_neg, counts_zero, counts_pos, labels=['-1','0','1'], colors=['mediumslateblue','grey','gold'], alpha=0.4),
    ax2.set_ylabel("Node state distribution (%)")
    ax2.set_ylim(0, 100)
    fig.legend(loc='outside upper left')

    plt.title(f"Magnetization and node state distribution of {topology_name} topology \n random initalisation with β = {beta}")
    plt.grid(True, alpha=0.3)
 
    filename = f"results/random_{topology_name.replace('-', '_')}_{param_name}.png"
    plt.savefig(filename)
    plt.show()

def simulation_no_external_field(adj_mat, topology_name, param_name, beta, l=0):
    #initialisation 
    
    mags = []
    for i in range(10):
        random_init = jax.random.randint(shape=(len(adj_mat),),minval=-1,maxval=2,key=jax.random.PRNGKey(np.random.randint(low=0, high=100000)),)
        network = ising_efficient.BeliefNetwork(
            sparse_adj=adj_mat,
            external_field=lambda t, node_idx: 0,
            init_state=np.copy(random_init),
            µ=µ,
            beta=beta,
        )
        result = network.run_for_steps(250, seed=i)

        magnetization_erdos = np.mean(result, axis=1)
        mags.append(np.copy(magnetization_erdos))
    mags_array = np.array(mags)

    plt.figure(figsize=(15,9))
    for m in mags_array: 
        plt.plot(m)

    plt.xlabel("Time steps")
    plt.ylabel("Magnetization")
    plt.ylim(-1,1)
    plt.title(f"Impact of {topology_name} topology on belief alignment \n random initialisation with β = {beta}")
    plt.grid(True, alpha=0.3)
    #plt.legend()
    filename = f"results/random_multiple_{topology_name.replace('-', '_')}_{param_name}.png"
    plt.savefig(filename)
    plt.show()
