import numpy as np
from functions import *
from simulatedannealing import SimulatedAnnealing
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

### --- 2D visualisation --- ###
# vid_folder = os.getcwd() + "/SAvideo"
# chain_folder = os.getcwd() + "/SAchain"
# strat = SimulatedAnnealing(2, 10, 1000, 500, seed=0, record_history=False)

# fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
# ax = fig.axes[0]
# strat.anneal(record_video=False, record_trajectory=True, chain_folder=chain_folder, vid_folder=vid_folder, fig=fig)

### --- convergence in value --- ###
seed = 30
strat = SimulatedAnnealing(dimension=6,
	 shotgun_samps=100, initial_samps=500, L=450, 
	 seed=seed, alpha=0.85, max_step=100., min_step=5., suppress_output=True)

best_value, mean_value, _ = strat.anneal()

plt.plot(np.arange(len(best_value)), best_value, label='Current Value')
plt.plot(np.arange(len(best_value)), mean_value, label='Archive Best')
plt.xlabel('Evalutations')
plt.legend()
plt.show()

### --- 6D performance --- ###
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set_xlabel('Evaluations')
# ax.set_ylabel('trace(D)')
# # for alpha in np.array([0.85]):
# # for alpha in np.array([0.5, 0.75, 0.85, 0.95, 0.99]):
# # for maxstep in np.array([25, 50, 100, 150, 200]):
# # for maxstep in np.array([100, 150, 200]):
# # for L in np.array([50, 150, 250, 350, 450]):
# for L in np.array([550, 650, 750]):
# 	obj_over_seeds = []
# 	iters_over_seeds = []
# 	# seeds = np.arange(1)
# 	seeds = np.arange(0, 50)
# 	for seed in tqdm(seeds):
# 		strat = SimulatedAnnealing(dimension=6,
# 		 shotgun_samps=1000, initial_samps=1000, L=L, 
# 		 seed=seed, alpha=0.85, max_step=100, min_step=5., suppress_output=True)
# 		_, _, iterations = strat.anneal(record_video=False)
# 		obj_over_seeds.append(strat.archive.get_current_optimum().objective)
# 		iters_over_seeds.append(iterations)

# 	obj_over_seeds = np.array(obj_over_seeds)
# 	print("L: "+str(L))
# 	print("best optimum found: "+str(np.min(obj_over_seeds)))
# 	print("mean optimum found: "+str(np.mean(obj_over_seeds)))
# 	print("variance of solutions: "+str(np.sqrt(np.var(obj_over_seeds))))
# 	print("mean number of iterations: "+str(np.mean(iters_over_seeds)))
# 	# ax.plot(np.arange(len(Dtrace)), 1./(6.*maxstep) * np.array(Dtrace), label='Max: {}'.format(str(maxstep)))
# fig.legend()
# plt.show()