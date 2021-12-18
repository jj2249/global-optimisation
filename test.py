from simulatedannealing import SimulatedAnnealing
from evolutionstrategy import EvolutionStrategy
from time import time


"""
This script will run an example of each algorithm and return the results plus the optimum.
The results are variable due to the stochastics nature of these algorithms.
"""


seed = 0


### --- Simulated Annealing --- ###
sa_start = time()
sa = SimulatedAnnealing(dimension=6,
	 shotgun_samps=100, initial_samps=500, L=450, 
	 seed=seed, alpha=0.85, max_step=100, min_step=5., suppress_output=True)
sa.anneal(record_video=False)
sa_time = time() - sa_start


### --- Evolution Strategy --- ###
es_start = time()
es = EvolutionStrategy(dimension=6,
	nparents=20, noffspring=140, epsilon=10., seed=seed)
while not es.converged:
	es.survival(record_parents=False)
	es.check_convergence()
	if not es.converged:
		es.mutate_offspring()
		es.recombine()
es_time = time() - es_start


### --- Output --- ###
print('\n')
print(10*'-'+"Simulated Annealing"+10*'-')
print('Runtime: '+str(sa_time)+'(s)')
print(sa)
print(30*'-')
print(2*'\n')
print(10*'-'+"Evolution Strategy"+10*'-')
print('Runtime: '+str(es_time)+'(s)')
print(es)
print(30*'-')
