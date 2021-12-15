import numpy as np
from functions import *
from solution import tsSolution
import matplotlib.pyplot as plt


class TabuSearch:
	def __init__(self, dimension, shotgun_samps, delta, N, M, record_history=False):
		self.dimension = dimension
		self.shotgun_samps = shotgun_samps
		self.startpoint = self.shotgun_initialisation()
		self.basepoint = self.startpoint.copy_solution()
		self.currentpoint = self.startpoint.copy_solution()
		self.delta = delta

		self.record_history = record_history
		if self.record_history:
			self.history = [self.startpoint]

		# short and medium term memory
		self.N = N
		self.M = M
		self.stm = []
		self.mtm = []

	def __repr__(self):
		return str(10*'-'+"Starting point"+10*'-'+'\n'+str(self.startpoint.__repr__()))

	
	
	def shotgun_initialisation(self):
		best_solution = None
		best_objective = np.inf
		# shoot a spread of points at the domain and keep only the best one
		for i in range(self.shotgun_samps):
			x = 500*np.random.randn(self.dimension)
			# reject until in the feasible region
			while not (np.all(x <=512.) and np.all(x >= -512)):
				x = 500*np.random.randn(self.dimension)
			# candidate solution in the feasible region
			candidate = tsSolution(x)
			if candidate.objective < best_objective:
				best_solution = candidate.copy_solution()
				best_objective = best_solution.objective

		return best_solution.copy_solution()


	def find_best_allowed_move(self):
		"""
		Increment each coordinate in turn to find the best move not in short term memory
		"""
		best_solution = None
		best_objective = np.inf
		for i in range(self.dimension):
			# upwards and downwards increments
			move_up =  self.currentpoint.copy_solution().coords
			move_down = self.currentpoint.copy_solution().coords
			move_up[i] += self.delta
			move_down[i] -= self.delta

			# check short term memory before evaluating objectives
			up_valid = True
			down_valid = True
			for prevpoint in self.stm:
				if np.all(move_up == prevpoint.coords):
					up_valid = False
				if np.all(move_down == prevpoint.coords):
					down_valid = False

			## if both moves are valid, compare the better of the two to the objective
			if up_valid and down_valid:
				up_sol = tsSolution(move_up)
				down_sol = tsSolution(move_down)
				if up_sol.objective < down_sol.objective:
					if up_sol.objective < best_objective:
						best_solution = up_sol.copy_solution()
						best_objective = up_sol.objective
				else:
					if down_sol.objective < best_objective:
						best_solution = down_sol.copy_solution()
						best_objective = down_sol.objective

			# if only one is valid, then compare it to the current best
			elif up_valid and (not down_valid):
				up_sol = tsSolution(move_up)
				if up_sol.objective < best_objective:
					best_solution = up_sol.copy_solution()
					best_objective = up_sol.objective

			elif (not up_valid) and down_valid:
				down_sol = tsSolution(move_down)
				if down_sol.objective < best_objective:
					best_solution = down_sol.copy_solution()
					best_objective = down_sol.objective

			# do nothing if neither are valid
		return best_solution.copy_solution()


	def check_pattern_move(self):
		if self.currentpoint.objective < self.basepoint.objective:
			# rp = rc + (rc - rb) i.e move by the base-current vector
			pattern_candidate_coords = np.array(2*(self.currentpoint.coords) - self.basepoint.coords)
			pattern_candidate = tsSolution(pattern_candidate_coords)
			# if the patterm move improves the objective further then keep it
			if pattern_candidate.objective < self.currentpoint.objective:
				return pattern_candidate.copy_solution()
			else:
				return self.currentpoint.copy_solution()
		else:
			return self.currentpoint.copy_solution()

	def make_move(self):
		self.currentpoint = self.find_best_allowed_move().copy_solution()
		# print(self.currentpoint)
		# print(self.basepoint)
		self.basepoint = self.check_pattern_move()
		self.stm.append(self.basepoint)
		if len(self.stm) > self.N:
			self.stm.pop(0)
		# print(10*'-'+'currentpoint'+10*'-')
		# print(self.currentpoint)
		# print(10*'-'+'memory'+10*'-')
		# print(self.stm)
		# print('\n')
		if self.record_history:
			self.history.append(self.basepoint)
		self.currentpoint = self.basepoint.copy_solution()


	def plot_history_on_contour(self):
			"""
			Plot the offspring on the contour map
			"""
			fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
			ax = fig.axes[0]
			for point in self.history:
				ax.scatter(point.coords[0], point.coords[1])
			return fig
				
			