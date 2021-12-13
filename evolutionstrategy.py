import numpy as np
from functions import *
from solution import esSolution
import matplotlib.pyplot as plt


class EvolutionStrategy:
	def __init__(self, dimension, nparents, noffspring, epsilon, seed):
		self.seed = seed
		self.dimension = dimension
		self.nparents = nparents
		self.noffspring = noffspring
		self.epsilon = epsilon
		self.converged = False
		# Generate an initial population
		self.offspring = None
		self.generate_offspring()
		self.parents = None
		np.random.seed(self.seed)


	def __repr__(self):
		"""
		Print method displays the best solution found (for now)
		"""
		objectives = np.array([offspring.objective for offspring in self.offspring])
		idx = np.argmin(objectives)
		return "Best objective: "+str(objectives[idx])+'\nCoordinates: '+str(self.offspring[idx].coords)


	def generate_offspring(self):
		"""
		Generate a set of feasible offspring
		"""
		offspring = []
		for _ in range(self.noffspring):
			x = 500*np.random.randn(self.dimension)
			# reject until in the feasible region
			while not (np.all(x <=512.) and np.all(x >= -512)):
				x = np.random.randn(self.dimension)
			offspring.append(esSolution(x))
		self.offspring = np.array(offspring)


	def survival(self):
		"""
		Select the best solutions (lowest objective) and assign as parents
		"""
		# collect offspring objectives for sorting
		objectivef = np.array([off.objective for off in self.offspring])
		# sort the offspring in increasing objective order (best to worst)
		idx = np.argsort(objectivef)
		strongest = np.take(self.offspring, idx)
		# assign the the new parents
		self.parents = strongest[:self.nparents]


	def recombine(self):
		"""
		Generate new offspring using discrete recombination for the coordinates and intermediate for the (co)variances
		"""
		new_offspring = []

		for _ in range(self.noffspring):
			# select 2 parents randomly
			p1c = np.random.randint(0, self.nparents)
			p2c = np.random.randint(0, self.nparents)

			# child coordinates determined by discrete recombination
			child = np.zeros(self.dimension)
			for i in range(self.dimension):
				# unbiased coin toss to determine which parent
				if np.random.uniform() > 0.5:
					child[i] = self.parents[p2c].coords[i]
				else:
					child[i] = self.parents[p1c].coords[i]

			# need to ensure positive definiteness of covariance matrix
			is_pos_def = False
			while not is_pos_def:
				# resample parents until a valid configuration found
				p1s = np.random.randint(0, self.nparents)
				p2s = np.random.randint(0, self.nparents)
				# new strategy parameters are the weighted sum of the two parents
				weight = 0.5
				new_variances = weight*self.parents[p1s].variances + (1-weight)*self.parents[p2s].variances
				new_rotangles = weight*self.parents[p1s].rotangles + (1-weight)*self.parents[p2s].rotangles
				new_child = esSolution(child, new_variances, new_rotangles)
				try:
					C = new_child.covariance_matrix()
					Cchol = np.linalg.cholesky(C+CHOL_FAC*np.eye(self.dimension))
					is_pos_def = True
				except np.linalg.LinAlgError:
					pass

			new_offspring.append(new_child)

		# store the new set of offspring
		self.offspring = np.array(new_offspring)

		# update each solutions objective
		for offspring in self.offspring:
			offspring.evaluate_objective()


	def mutate_offspring(self):
		"""
		Mutate each of the offspring
		"""
		for offspring in self.offspring:
			offspring.mutate()


	def check_convergence(self, suppress=True):
		"""
		Compare current best and worst parents to determine convergence
		Can opt to not suppress output for a convergence readout
		"""
		objectives = np.sort(np.array([parent.objective for parent in self.parents]))
		# print(objectives)
		diff = np.abs(objectives[0]-objectives[-1])
		if not suppress:
			print("Absolute difference: "+str(diff))
		# converged once absolute difference is less than user defined limit
		if diff < self.epsilon:
			self.converged = True


	def best_objective(self):
		"""
		Return the best objective value
		"""
		objectives = np.sort(np.array([offspring.objective for offspring in self.offspring]))
		return objectives[0]


	def mean_objective(self):
		"""
		Return the mean objective value of parents
		"""
		objectives = np.array([parent.objective for parent in self.parents])
		return np.mean(objectives, axis=0)


	def plot_parents_on_contour(self):
		"""
		Plot the offspring on the contour map
		"""
		fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
		ax = fig.axes[0]
		for parent in self.parents:
			ax.scatter(parent.coords[0], parent.coords[1])
		return fig


	def plot_parents_on_surface(self):
		"""
		Plot the offspring on the surface
		"""
		fig = plot_eggholder(2*512, ThreeD=True, Contour=False)
		ax = fig.axes[0]
		for parent in self.parents:
			ax.scatter(parent.coords[0], parent.coords[1], parent.objective)
		return fig