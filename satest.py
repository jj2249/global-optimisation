import numpy as np
from functions import *
from simulatedannealing import SimulatedAnnealing
from tqdm import tqdm
import matplotlib.pyplot as plt


# vid_folder = os.getcwd() + "/SAvideo"

# strat = SimulatedAnnealing(6, 10, 100, 500, record_history=False)
# print(strat)
# fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
# ax = fig.axes[0]
# strat.anneal(record_video=False, vid_folder=vid_folder, fig=fig)
# print(strat)
# # strat.plot_current_on_contour()
# print(strat.archive)
# # plt.show()
# # os.chdir("SAvideo")
# # os.system("ffmpeg -r 10 -i file%02d.png -vcodec mpeg4 -y movie.mp4")

obj_over_seeds = []
# seeds = np.arange(1)
seeds = np.arange(0, 5)
for seed in tqdm(seeds):
	strat = SimulatedAnnealing(6, 10, 100, 500, seed, suppress_output=True)
	strat.anneal(record_video=False)
	obj_over_seeds.append(strat.archive.get_current_optimum().objective)

obj_over_seeds = np.array(obj_over_seeds)
print("best optimum found: "+str(np.min(obj_over_seeds)))
print("mean optimum found: "+str(np.mean(obj_over_seeds)))
print("variance of solutions: "+str(np.var(obj_over_seeds)))