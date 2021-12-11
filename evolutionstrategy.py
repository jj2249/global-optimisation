import numpy as np
from functions import eggholder, arctan, plot_eggholder
import matplotlib.pyplot as plt


class Solution:
	def __init__(self, coords):
		self.coords = coords
		self.dim = self.coords.shape[0]
		self.npairs = int(self.dim*(self.dim-1)/2.)
		self.variances = 1*np.ones(self.dim)
		self.rotangles = np.zeros(self.npairs)
		self.tau = 1./np.sqrt(2*np.sqrt(self.dim))
		self.taup = 1./np.sqrt(2*self.dim)
		self.beta = 0.0873
		self.objective = None
		self.evaluate_objective()


	def __repr__(self):
		return str("Solution coordinates: "+str(self.coords)+'\n'+
			"Variances: "+str(self.variances)+'\n'+
			"Rotation angles: "+str(self.rotangles)+'\n'+
			"Objective: "+str(self.objective))



	def evaluate_objective(self):
		self.objective = eggholder(self.coords)


	def covars_from_rot(self):
		"""
		Return a (minimal) set of covariances from the corresponding rotation angles and variances
		"""
		covars = np.zeros(self.npairs)
		# need to index the covariance variables separately from the variances
		count = 0
		for i in range(self.dim-1):
			# exploit symmetry
			for j in range(i+1, self.dim):
				covars[count] = 0.5*(self.variances[i]-self.variances[j])*np.tan(2*self.rotangles[count])
				count += 1
		return covars


	def covariance_matrix(self):
		"""
		Generate the covariance matrix from the current strategy parameters
		"""
		covars = self.covars_from_rot()
		C = np.diag(self.variances)

		# this took way too long to work out
		count = self.npairs
		count2 = self.npairs-self.dim+1
		countp = self.dim-1
		countp2 = self.dim-2

		# build in a top-right to bottom-left fashion
		for i in range(self.dim-1):
			C[i, i+1:] = covars[self.npairs-count:self.npairs-count2]
			C[i+1:, i] = covars[self.npairs-count:self.npairs-count2]

			count -= countp
			count2 -= countp2
			countp -= 1
			countp2 -= 1
		return C


	def visualise_covariance(self):
		"""Visualise the covariance as a heatmap"""
		C = self.covariance_matrix()
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.imshow(C)
		return fig


	def mutate(self):
		"""
		Mutate the control variables by randomly sampling from a multivariate normal dist
		"""
		# random samples for mutation of strategy parameters
		aleph0 = np.random.randn()*np.ones(self.dim)
		alephis = np.random.randn(self.dim)
		alephijs = np.random.randn(self.npairs)
		# mutate the strategy parameters
		self.variances = self.variances * np.exp(self.taup*aleph0+self.tau*alephis)
		self.rotangles = self.rotangles + self.beta*alephijs

		C = self.covariance_matrix()
		# need to work on stability here?
		Cchol = np.linalg.cholesky(C + 1e-4*np.eye(self.dim))

		self.coords = self.coords + Cchol @ np.random.randn(self.dim)
		while not (np.all(self.coords <=512.) and np.all(self.coords >= -512)):
			self.coords = self.coords + Cchol @ np.random.randn(self.dim)



class EvolutionStrategy:
	def __init__(self, dimension, nparents, noffspring, epsilon):
		self.dimension = dimension
		self.nparents = nparents
		self.noffspring = noffspring
		self.epsilon = epsilon
		# Generate an initial population
		self.offspring = None
		self.generate_offspring()
		self.parents = None


	def __repr__(self):
		"""
		Print method displays the best solution found (for now)
		"""
		objectives = np.array([offspring.objective for offspring in self.offspring])
		idx = np.argmax(objectives)
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
			offspring.append(Solution(x))
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


	def discrete_recombination(self):
		"""
		Generate new offspring using the discrete pair recombination method
		"""
		new_offspring = []

		for _ in range(self.noffspring):
			# select 2 parents randomly
			p1 = np.random.randint(0, self.nparents)
			p2 = np.random.randint(0, self.nparents)
			child = np.zeros(self.dimension)

			for i in range(self.dimension):
				# unbiased coin toss to determine which parent
				if np.random.uniform() > 0.5:
					child[i] = self.parents[p2].coords[i]
				else:
					child[i] = self.parents[p1].coords[i]
			
			new_offspring.append(Solution(child))

		# store the new set of offspring
		self.offspring = np.array(new_offspring)

		# update each solutions objective
		for offspring in self.offspring:
			offspring.evaluate_objective()



	def intermediate_recombination(self):
		"""
		Generate new offspring using the intermediate pair recombination method
		"""
		new_offspring = []

		for _ in range(self.noffspring):
			# select 2 parents randomly
			p1 = np.random.randint(0, self.nparents)
			p2 = np.random.randint(0, self.nparents)
			# new offspring is the mean of these two parents
			new_offspring.append(Solution(0.5*(self.parents[p1].coords+self.parents[p2].coords)))

		# update optimiser class member variable
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


	def plot_population(self):
		"""
		"""
		fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
		ax = fig.axes[0]
		# ax = fig.add_subplot()
		# ax.set_xlim(-600, 600)
		# ax.set_ylim(-600, 600)
		for offspring in self.offspring:
			ax.scatter(offspring.coords[0], offspring.coords[1])
		return fig