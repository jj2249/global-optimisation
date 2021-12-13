from functions import *
from time import time
from tqdm import tqdm
from evolutionstrategy import EvolutionStrategy
from archive import Archive

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
# plt.show()

# sol = Solution(np.random.randn(6))
# print("run1")
# print(sol)
# print("\n\n\n")
# sol.mutate()
# print("run2")
# print(sol)
# print("\n\n\n")
# sol.visualise_covariance()
# plt.show()

# seeds = np.arange(1, 40)
# iterations = []
# best_objs = []
# mean_objs = []

# for seed in tqdm(seeds):
# 	strat = EvolutionStrategy(2, 20, 140, 10., seed)
# 	count = 0
# 	while not strat.converged:
# 		strat.survival()
# 		strat.evaluate_convergence()
# 		if not strat.converged:
# 			strat.mutate_offspring()
# 			strat.recombine()
# 		count += 1
# 	iterations.append(count)
# 	best_objs.append(strat.best_objective())
# 	mean_objs.append(strat.mean_objective())


# seed = 1
# iterations = 0
# best_objs = []
# mean_objs = []

# strat = EvolutionStrategy(2, 20, 100, 10., seed)
# while not strat.converged:
# 	strat.survival()
# 	strat.check_convergence()
# 	if not strat.converged:
# 		strat.mutate_offspring()
# 		strat.recombine()
# 	iterations += 1
# 	best_objs.append(strat.best_objective())
# 	mean_objs.append(strat.mean_objective())

# print(strat)

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.plot(np.arange(iterations), mean_objs, label='mean')
# ax2 = ax1.twinx()
# ax2.plot(np.arange(iterations), best_objs, color='orange', label='best')
# ax1.set_ylabel('mean')
# ax2.set_ylabel('best')
# ax1.set_ylim(bottom=-1000)
# ax2.set_ylim(bottom=-1000)
# fig.legend()
# fig.show()
# plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.plot(np.arange(iterations), mean_objs, label='mean')
# ax1.plot(np.arange(iterations), best_objs, color='orange', label='best')
# fig.legend()
# fig.show()
# plt.show()

archive = Archive(30, 10., 0.1)

obj_over_seeds = []
seeds = np.arange(1)
# seeds = np.arange(0, 51)
for seed in tqdm(seeds):
	strat = EvolutionStrategy(2, 20, 140, 10., seed)
	while not strat.converged:
		strat.survival()
		for parent in strat.parents:
			archive.check_candidate(parent)
		strat.check_convergence()
		if not strat.converged:
			strat.mutate_offspring()
			strat.recombine()
	obj_over_seeds.append(strat.best_objective())

obj_over_seeds = np.array(obj_over_seeds)
print(obj_over_seeds)
print("mean optimum found: "+str(np.mean(obj_over_seeds)))
print("variance of solutions: "+str(np.var(obj_over_seeds)))
print(archive)
archive.plot_on_contour()
plt.show()