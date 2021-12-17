from functions import *
from time import time
from tqdm import tqdm
from evolutionstrategy import EvolutionStrategy
from archive import Archive
import os
import matplotlib.pyplot as plt

### --- 2D visualisation --- ###

# folder = os.getcwd() + "/ESpops"

# seed = 1

# strat = EvolutionStrategy(2, 20, 140, 10., seed)
# while not strat.converged:
# 	strat.survival(record_parents=True, folder=folder)
# 	strat.check_convergence()
# 	if not strat.converged:
# 		strat.mutate_offspring()
# 		strat.recombine()
# fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
# ax = fig.axes[0]


### --- convergence in value --- ###
seed = 30
strat = EvolutionStrategy(dimension=6,
	nparents=20, noffspring=140, epsilon=10., seed=seed)

best_value = []
mean_value = []
best_value1 = []
mean_value1 = []

while not strat.converged:
		strat.survival(record_parents=False)
		strat.check_convergence()
		best_value1.append(strat.best_objective())
		mean_value1.append(strat.mean_objective())
		best_value.append(strat.archive.get_current_optimum().objective)
		mean_value.append(strat.archive.get_mean_archived_solution())
		if not strat.converged:
			strat.mutate_offspring()
			strat.recombine()

plt.plot(np.arange(len(best_value)), best_value, label='Archived Best')
plt.plot(np.arange(len(best_value)), mean_value, label='Archived Mean')
plt.plot(np.arange(len(best_value)), best_value1, label='Current Best')
plt.plot(np.arange(len(best_value)), mean_value1, label='Current Mean')
plt.xlabel('Generation')
plt.legend()
plt.show()

### --- 6D performance --- ###
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set_xlabel('Generation')
# ax.set_ylabel('Mean parent objective')
# # for noffspring in np.array([140]):
# for noffspring in np.array([40, 60, 80, 100, 120, 140, 160, 180, 200]):
# 	obj_over_seeds = []
# 	no_generations = []
# 	# seeds = np.arange(1)
# 	seeds = np.arange(0, 50)
# 	for seed in tqdm(seeds):
# 		best_obj_in_parents = []
# 		generations = 0
# 		strat = EvolutionStrategy(dimension=6,
# 		nparents=20, noffspring=noffspring, epsilon=10., seed=seed)
# 		while not strat.converged:
# 			strat.survival(record_parents=False)
# 			strat.check_convergence()
# 			best_obj_in_parents.append(strat.mean_objective())
# 			if not strat.converged:
# 				strat.mutate_offspring()
# 				# strat.recombine()
# 				strat.global_recombine()
# 				generations += 1
# 		# ax.plot(np.arange(len(best_obj_in_parents)), best_obj_in_parents)
# 		obj_over_seeds.append(strat.archive.get_current_optimum().objective)
# 		no_generations.append(generations)

# 	obj_over_seeds = np.array(obj_over_seeds)
# 	print("n offspring: "+str(noffspring))
# 	print("best optimum found: "+str(np.min(obj_over_seeds)))
# 	print("mean optimum found: "+str(np.mean(obj_over_seeds)))
# 	print("std dev of solutions: "+str(np.sqrt(np.var(obj_over_seeds))))
# 	print("no of generations: "+str(np.mean(no_generations)))
# # plt.show()