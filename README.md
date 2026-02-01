# Universiteit van Amsterdam - Complex System Simulation 2026

This is the repo that holds code to reproduce all of our results and figures.

The simulator itself cannot use any `assert` statements, as these cannot be compiled into JAX code, however, we do use `assert` statements in other reused library code, and test some code with `pytest`, see below.

## Installation & Setup
<pre>
git clone [git@github.com:Whatisthisname/ComplexSystem.git]
pip install -r requirements.txt
</pre>

## Code and what it does

`clustering_coefficient.py` holds the functions used to approximate the clustering coefficient in a big graph.

`gen_images_of_graphs.py` can be executed to generate images of 3 networks that show the different topologies that we experiment with in this project. This was purely for the slides.

`ising_efficient.py` holds our efficient JAX implementation of the 3-state Potts model with the time-varying external field. It's a class that can be instantiated and then `.run_for_steps(*)` to store results.

`network_generation.py` holds 3 functions that each take some network parameters and create a compressed-sparse-row format adjacency matrix for different network topologies (erdos-renyi, scale-free, small-world) that we use in the simulator.

`plot_clustering.py` can be executed to create plots for the clustering coefficient experiments from the appendix in the slides.

`run_data_collection.py` can be executed to run the simulation for a range of parameters. The results are then stored in the `results/lambda_walk/` directory, for later processing.

`sparse_adj_mat.py` holds the definition of the `sparse_adj_mat.Sparse_Adjacency_Matrix` class that is just the implementation and helper functions for the compressed-sparse-row format for adjacency matrices.

`test_clustering_coefficient.py` uses `pytest` to test the implementation of the approximate and exact clustering coefficient calculation algorithm.

`topology_simulation.py` (used in `network_topology.ipynb`) and `heatmaps/*.py` hold the functions used for the main experiments, hypothesis 2 and 3. The `heatmaps/*.py` files generate the voronoi plots from 2D Halton sequences of points. These files also use the `.pkl` files like `opinion_change_threshold_erdos.pkl` and `memory_vs_fraction_temp_1.5_v4.pkl` to store results and to refine and add more points to increase resolution if desired. `topology_simulation.py` is plotting, for given parameter sets, simulation trajectories and animating them.

`bifurcation/bifurcation.py` has code to generate the bifurcation temperature sweep, while `bifurcation/bifurcation.ipynb` is a convenience notebook to play around with plotting parameters after the data has been collected.

`use_network_example.py` is what the name says, it imports the `ising_efficient.py` library and uses the simulator to give a minimal runnable example.

`voronoi.py` has code to plot the voronoi diagrams. Partially it is code from stackoverflow that we have cited in the file as well.

`diameter.ipynb` can compute the diameter of a network, where we also group by belief beforehand.

`project_report.ipynb` was our intermediate results-gathering place.

`gif_generation.ipynb`, well, generates GIFs.

# Reproducing Our Results
To replicate the findings of this study, execute the scripts in the following order. 
### 1. Phase Transitions (Bifurcation)
* Run: python bifurcation/bifurcation.py
* Analyze: Use bifurcation/bifurcation.ipynb to visualize the steady-state magnetization
### 2. Voronoi Parameter Search 
see how different topologies (Erdős-Rényi, Small-World, Scale-Free) respond to propaganda events (Hypothesis 2)
* Run: python heatmaps/*.py
### 3. Dynamics & GIF Generation
* Run: python topology_simulation.py and use the gif_generation.ipynb notebook. This produces figures saved in the results folder.
### 4. Memory Coefficient analysis (Hypothesis 3)
* Run: Execute the memory-specific Voronoi scripts in the heatmaps/ directory.
### 5. Network Structure (Clustering Coefficients)
* Run: python plot_clustering.py - approximate clustering coefficients over time to see how components merge or fragment during exposure.
