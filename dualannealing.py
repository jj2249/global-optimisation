from functions import *
from scipy.optimize import dual_annealing
from time import time
from tqdm import tqdm
from simulatedannealing import SimulatedAnnealing
from evolutionstrategy import EvolutionStrategy

bounds = np.array([[-512, 512],
	[-512, 512],
	[-512, 512],
	[-512, 512],
	[-512, 512],
	[-512, 512]])

da_obj = []
sa_obj = []
es_obj = []

da_tim = []
sa_tim = []
es_tim = []
for seed in (np.arange(0, 50)):
	da = dual_annealing(eggholder, bounds, maxiter=15000, seed=seed)
	print(da.fun)

# for seed in tqdm(np.arange(0, 10)):
# 	da_start = time()
# 	da = dual_annealing(eggholder, bounds, maxiter=15000, seed=seed)
# 	da_tim.append(time()-da_start)
# 	da_obj.append(da.fun)

# 	sa_start = time()
# 	sa = SimulatedAnnealing(dimension=6,
# 		 shotgun_samps=100, initial_samps=500, L=450, 
# 		 seed=seed, alpha=0.85, max_step=100, min_step=5., suppress_output=True)
# 	sa.anneal(record_video=False)
# 	sa_tim.append(time()-sa_start)
# 	sa_obj.append(sa.archive.get_current_optimum().objective)

# 	es_start = time()
# 	es = EvolutionStrategy(dimension=6,
# 		nparents=20, noffspring=140, epsilon=10., seed=seed)
# 	while not es.converged:
# 		es.survival(record_parents=False)
# 		es.check_convergence()
# 		if not es.converged:
# 			es.mutate_offspring()
# 			es.recombine()
# 	es_tim.append(time()-es_start)
# 	es_obj.append(es.archive.get_current_optimum().objective)