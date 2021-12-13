import numpy as np
from functions import *
import matplotlib.pyplot as plt


class Solution:
	def __init__(self, coords):
		self.coords = coords
		self.dim = self.coords.shape[0]
		self.npairs = int(self.dim*(self.dim-1)/2.)
		self.objective = None
		self.evaluate_objective()


	def __repr__(self):
		return str("Solution coordinates: "+str(self.coords)+'\n'+
			"Variances: "+str(self.variances)+'\n'+
			"Rotation angles: "+str(self.rotangles)+'\n'+
			"Objective: "+str(self.objective))


	def evaluate_objective(self):
		self.objective = eggholder(self.coords)


class esSolution(Solution):
	def __init__(self, coords, variances=None, rotangles=None):
		Solution.__init__(self, coords)
		if variances is None:
			self.variances = 100*np.ones(self.dim)
		else:
			self.variances = variances
		if rotangles is None:
			self.rotangles = np.zeros(self.npairs)
		else:
			self.rotangles = rotangles
		self.tau = 1./np.sqrt(2*np.sqrt(self.dim))
		self.taup = 1./np.sqrt(2*self.dim)
		self.beta = 0.0873


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

			# 2-fold counting scheme
			count -= countp
			count2 -= countp2
			countp -= 1
			countp2 -= 1
		return C


	def visualise_covariance(self):
		"""
		Visualise the covariance as a heatmap
		"""
		C = self.covariance_matrix()
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.imshow(C)
		return fig


	def mutate(self):
		"""
		Mutate the control variables by randomly sampling from a multivariate normal dist
		"""
		# need to ensure positive definiteness of the covariance matrix
		is_pos_def = False
		while not is_pos_def:
			# random samples for mutation of strategy parameters
			aleph0 = np.random.randn()*np.ones(self.dim)
			alephis = np.random.randn(self.dim)
			alephijs = np.random.randn(self.npairs)
			# mutate the strategy parameters
			candidate_variances = self.variances * np.exp(self.taup*aleph0+self.tau*alephis)
			candidate_rotangles = self.rotangles + self.beta*alephijs

			C = self.covariance_matrix()
			try:
				# repeat process until positive definiteness achieved
				Cchol = np.linalg.cholesky(C + CHOL_FAC*np.eye(self.dim))
				is_pos_def = True
			except:
				pass

		self.variances = candidate_variances
		self.rotangles = candidate_variances

		# mutate coords using a multivariate normal
		candidate_coords = self.coords + Cchol @ np.random.randn(self.dim)
		
		# need to check feasibility of new coordinates
		tries = 0
		while not (np.all(candidate_coords <=512.) and np.all(candidate_coords >= -512)):
			if tries < 20:
				candidate_coords = self.coords + Cchol @ np.random.randn(self.dim)
			else:
				candidate_coords = self.coords
			tries+=1
		self.coords = candidate_coords
