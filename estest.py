from functions import *
from time import time
from tqdm import tqdm
from evolutionstrategy import EvolutionStrategy
from archive import Archive

### --- Quick implementation of Eggholder function --- ###
# plot_eggholder(2*512, ThreeD=True, Contour=True)
# plt.show()

# evals = 15000
# Dvals = np.arange(6, 206, 20)

# naivetimes = []
# bettertimes = []

# for D in tqdm(Dvals):
# 	x = np.random.randn(D)
# 	start1 = time()
# 	for _ in range(evals):
# 		eggholder_naive(x)
# 	naivetimes.append(time()-start1)

# 	start2 = time()
# 	for _ in range(evals):
# 		eggholder(x)
# 	bettertimes.append(time()-start2)

# plt.plot(Dvals, naivetimes, label='naive')
# plt.plot(Dvals, bettertimes, label='vectorised')


### --- Multiple seeds --- ###
# seeds = np.arange(1, 40)
# seeds = np.arange(1)
# iterations = []
# best_objs = []
# mean_objs = []

# for seed in tqdm(seeds):
# 	strat = EvolutionStrategy(2, 20, 140, 10., seed)
# 	for i in range(100):
# 		strat.survival()
# 		strat.mutate_offspring()
# 		strat.recombine()
# 	best_objs.append(strat.best_objective())
# 	mean_objs.append(strat.mean_objective())

# for seed in tqdm(seeds):
# 	strat = EvolutionStrategy(2, 20, 140, 5., seed)
# 	count = 0
# 	while not strat.converged:
# 		strat.survival()
# 		strat.check_convergence()
# 		if not strat.converged:
# 			strat.mutate_offspring()
# 			strat.recombine()
# 		count += 1
# 	iterations.append(count)
# 	best_objs.append(strat.best_objective())
# 	mean_objs.append(strat.mean_objective())

# print(mean_objs)
# print(best_objs)
# print(count)
# strat.plot_parents_on_contour()
# strat.plot_parents_on_surface()

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.plot(np.arange(iterations), np.array(mean_objs), label='mean')
# ax1.plot(np.arange(iterations), np.array(best_objs), color='orange', label='best')
# fig.legend()
# fig.show()

### --- Archive --- ###


obj_over_seeds = []
# seeds = np.arange(1)
seeds = np.arange(0, 20)
for seed in tqdm(seeds):
	strat = EvolutionStrategy(6, 20, 140, 10., seed)
	while not strat.converged:
		strat.survival()
		strat.check_convergence()
		if not strat.converged:
			strat.mutate_offspring()
			strat.recombine()
	obj_over_seeds.append(strat.best_objective())

obj_over_seeds = np.array(obj_over_seeds)
print("best optimum found: "+str(np.min(obj_over_seeds)))
print("mean optimum found: "+str(np.mean(obj_over_seeds)))
print("variance of solutions: "+str(np.var(obj_over_seeds)))
# print(archive)
# archive.plot_on_contour()
# strat.plot_parents_on_contour()
# archive.plot_on_surface()
# plt.show()