from functions import *
from time import time
from tqdm import tqdm
from evolutionstrategy import Solution, EvolutionStrategy

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

strat = EvolutionStrategy(2, 20, 140)
print(strat)
strat.plot_population()
for i in tqdm(range(100)):
	strat.survival()
	strat.discrete_recombination()
	strat.mutate_offspring()

print(strat)
strat.plot_population()
plt.show()