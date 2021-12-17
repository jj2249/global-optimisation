from functions import eggholder, eggholder_naive
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from time import time


### --- Quick implementation of Eggholder function --- ###
# plot_eggholder(2*512, ThreeD=True, Contour=True)
# plt.show()

evals = 15000
Dvals = np.arange(6, 256, 25)

naivetimes = []
bettertimes = []

for D in tqdm(Dvals):
	x = np.random.randn(D)
	start1 = time()
	for _ in range(evals):
		eggholder_naive(x)
	naivetimes.append(time()-start1)

	start2 = time()
	for _ in range(20*evals):
		eggholder(x)
	bettertimes.append((time()-start2)/20)

fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
ax1.plot(Dvals, naivetimes, label='naive')
# ax1.plot(Dvals, bettertimes, label='vectorised', color='orange')
ax2.plot(Dvals, bettertimes, label='vectorised', color='orange')
fig.legend()
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Naive time (s)')
ax2.set_ylabel('Vectorised time (s)')
ax1.set_xlim(left=0)
plt.show()