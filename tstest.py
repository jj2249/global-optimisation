from functions import *
from time import time
from tqdm import tqdm
from tabusearch import TabuSearch
from archive import Archive

strat = TabuSearch(2, 10, delta=10., N=7, M=4, record_history=True)
for i in tqdm(range(100)):
	strat.make_move()
strat.plot_history_on_contour()
print(15*'-'+"New Run"+15*'-')
print(strat.history)
plt.show()
