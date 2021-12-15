import numpy as np
from functions import *
from solution import saSolution
import matplotlib.pyplot as plt
from tqdm import tqdm
from archive import Archive
from scipy import stats as ss


class SimulatedAnnealing:
	def __init__(self, dimension, shotgun_samps, initial_samps, L, seed, suppress_output=False, record_history=False):
		self.dimension = dimension
		# intial step size matrix
		self.D = 50*np.eye(self.dimension)
		self.R = None

		# parameters for updating D and T
		self.gamma = 0.1
		self.omega = 2.1
		self.alpha = 0.8

		# initialise chain using shotgun initialisation
		self.x = self.shotgun_initialisation(shotgun_samps)
		# initial temperature estimation
		self.T = self.estimate_initial_temp(0.8, initial_samps)
		self.suppress_output = suppress_output

		if not self.suppress_output:
			print(10*'-'+"Initial Temp"+10*'-'+'\n'+str(self.T))

		# chain length
		self.L = L
		# store solutions in the archive
		self.archive = Archive(30, 10., 0.1)

		# whether or not to store the entire history of the chain for plotting
		self.record_history = record_history
		if self.record_history:
			self.history = [self.x]

		# seed the rng
		self.seed = seed
		np.random.seed(seed)


	def __repr__(self):
		if self.record_history:
			return str(10*'-'+"Chain"+10*'-' +'\n'+
				str(self.history.__repr__()) +'\n'+
				10*'-'+"Current Optimum"+10*'-' +'\n'+
				str(self.archive.get_current_optimum().__repr__()))
		else:
			return str(10*'-'+"Current Value"+10*'-' +'\n'+
				str(self.x.__repr__()) +'\n'+
				10*'-'+"Current Optimum"+10*'-' +'\n'+
				str(self.archive.get_current_optimum().__repr__()))


	def shotgun_initialisation(self, shotgun_samps):
		"""
		Requires 1 objective evalution per sample used
		"""
		# placeholders
		best_solution = None
		best_objective = np.inf

		# shoot a spread of points at the domain and keep only the best one
		for i in range(shotgun_samps):
			# x = 500*np.random.randn(self.dimension)
			# # reject until in the feasible region
			# while not (np.all(x <=512.) and np.all(x >= -512)):
			# 	x = 500*np.random.randn(self.dimension)
			x = np.random.uniform(low=-512., high=512., size=self.dimension)
			# candidate solution in the feasible region
			candidate = saSolution(x)
			if candidate.objective < best_objective:
				best_solution = candidate
				best_objective = best_solution.objective

		return best_solution.copy_solution()


	def estimate_initial_temp(self, chi, initial_samps):
		"""
		Requires 1 objective evalution per sample used
		"""
		# store initial values for restarting purposes
		point = self.x
		D = self.D

		# needed to calculate the mean increase
		pos_sum = 0.
		num_pos = 0

		for i in range(initial_samps):
			# propose inital search direction
			u = np.random.uniform(low=-1., high=1., size=self.dimension)
			step = D @ u
			while not (np.all(point.coords+step<=512) and np.all(point.coords+step>=-512)):
				u = np.random.uniform(low=-1., high=1., size=self.dimension)
				step = D @ u

			# a measure of step size
			dhat = np.sqrt(np.sum(np.square(step)))

			# change in the objective
			new_obj = eggholder(point.coords+step)
			deltaf =  new_obj- point.objective
			
			if deltaf > 0.:
				# scale the increases
				pos_sum += (deltaf/dhat)
				num_pos += 1

			# move the chain (accept any move)
			new_coords = point.coords + step
			point = saSolution(new_coords, objective=new_obj)

			# update D
			R = np.diag(np.abs(step))
			D = (1-self.gamma)*D + self.gamma*self.omega*R

		# mean increase in the objective function
		deltafplus = pos_sum/num_pos
		# corresponding T for probability of accepting upwards move to be chi
		return -1*deltafplus / np.log(chi)

	# def increment_chain(self, acceptances, funcvals):
	def increment_chain(self, acceptances):
		"""
		Requires 1 objective evaluation per increment
		- acceptances tracks the number of accepted moves (used in the annealing schedule)
		"""
		# propose step
		u = np.random.uniform(low=-1., high=1., size=self.dimension)
		step = self.D @ u

		# step size measure
		dhat = np.sqrt(np.sum(np.square(step)))

		new_obj = eggholder(self.x.coords+step)
		# change in objective due to step
		deltaf = new_obj - self.x.objective

		# accept the step if it decreases the objective
		if deltaf < 0:
			new_coords = self.x.coords + step
			self.x = saSolution(new_coords, objective=new_obj)

			# update strategy parameters based on accepted step
			self.R = np.diag(np.abs(step))
			self.D = (1-self.gamma)*self.D + self.gamma*self.omega*self.R
			acceptances += 1
			# funcvals.append(self.x.objective)

		# otherwise, accept the move with probability p
		else:
			p = np.exp(-1*deltaf / (self.T*dhat))
			z = np.random.uniform()

			# biased coin toss for accepting
			if z < p:
				new_coords = self.x.coords + step
				self.x = saSolution(new_coords, objective=new_obj)

				# update strategy parameters if the move is accepted
				self.R = np.diag(np.abs(step))
				self.D = (1-self.gamma)*self.D + self.gamma*self.omega*self.R
				acceptances += 1
				# funcvals.append(self.x.objective)

		# add to chain if tracking the history
		if self.record_history:
			self.history.append(self.x)
		return acceptances#, funcvals


	def anneal(self, record_video=False, vid_folder=None, fig=None):
		"""
		Run the full annealing schedule
		"""
		# option to visualise path of chain
		if record_video:
			ax = fig.axes[0]

		# cap the total number of steps allowed in the full schedule
		total_iters = 0
		while (total_iters < 15000):
			# start of chain at a new temperature
			if not self.suppress_output:
				print(10*'-'+"Current Temp"+10*'-'+'\n'+str(self.T))
			# number of accepted moves at this temperature
			acceptances = 0
			# number of proposed moves at this temperature
			iters = 0
			# funcvals = []
			
			# run each chain until either a certain number of acceptances achieved or a certain number of proposals are made, whichever is first
			while (acceptances<0.6*self.L) and (iters<self.L):
				acceptances = self.increment_chain(acceptances)
				# acceptances, funcvals = self.increment_chain(acceptances, funcvals)

				# generate the plot for the video
				if record_video:
					point = ax.scatter(self.x.coords[0], self.x.coords[1], zorder=1)
					plt.savefig(vid_folder+"/file%02d"%iters)
					point.remove()

				iters += 1
				
				# check the current location as a candidate for archiving
				self.archive.check_candidate(self.x)

			# end of the chain at this temperature so record number of iterations
			total_iters += iters

			# self.alpha = max(0.5, np.exp(-0.7*self.T / np.sqrt(np.var(funcvals))))
			# print(self.alpha)
			
			# decrement temperature
			self.T = self.alpha * self.T
			# self.x = self.archive.get_current_optimum().copy_solution()


	def plot_history_on_contour(self):
			"""
			Plot the chain on the contour map
			"""
			fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
			ax = fig.axes[0]
			for point in self.history:
				ax.scatter(point.coords[0], point.coords[1])
			return fig

	def plot_current_on_contour(self):
			"""
			Plot the current point on the contour map
			"""
			fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
			ax = fig.axes[0]
			ax.scatter(self.x.coords[0], self.x.coords[1])
			return fig