from functions import *


class Archive:
	def __init__(self, L, Dmin, Dsim):
		# archive should store a set of Solution objects
		self.archived_solutions = []
		self.L = L
		self.Dmin = Dmin
		self.Dsim = Dsim
		self.archive_size = 0

	def __repr__(self):
		coords = np.array([archived.coords for archived in self.archived_solutions])
		objs = np.array([archived.objective for archived in self.archived_solutions])
		return str("Archived Solutions: "+str(coords)+'\n'+"Corresponding objectives: "+str(objs))

	def get_worst_archived_solution(self):
		"""
		Index and Objective of the worst currently archived solution
		"""
		objectives = np.array([archived.objective for archived in self.archived_solutions])
		idx = np.argmax(objectives)
		return idx, self.archived_solutions[idx].objective


	def get_best_archived_solution(self):
		"""
		Index and objective of the best currently archived solutions
		"""
		objectives = np.array([archived.objective for archived in self.archived_solutions])
		idx = np.argmin(objectives)
		return idx, self.archived_solutions[idx].objective


	def get_similar_idx(self, candidate):
		"""
		Check a candidate solution for similarity against the other archived solutions
		"""
		similar_idx = []
		closest_idx = None
		closest_distance = np.inf
		for i in range(self.archive_size):
			# check euclidian norm of the difference
			distance = np.sum(np.sqrt(np.linalg.norm(candidate.coords - self.archived_solutions[i].coords)))
			# if they are close, store index
			if distance < self.Dmin:
				similar_idx.append(i)
				# check if this is the most similar solution
				if distance < closest_distance:
					closest_idx = i
					closest_distance = distance
		return similar_idx, closest_idx, closest_distance


	def check_candidate(self, candidate):
		"""
		Incorporate a new candidate solution using the logic in the lecture notes
		"""
		similar_idx, closest_idx, closest_distance = self.get_similar_idx(candidate)
		# case where the archive is not full
		if self.archive_size < self.L:
			# check dissimilarity and append if criterion met
			if len(similar_idx) == 0:
				self.archived_solutions.append(candidate)
				self.archive_size += 1
		# if archive is full we need to check optimality
		else:
			# if it is dissimilar, include if it is better than the worst
			if len(similar_idx) == 0:
				# index of the worst solution
				idx, worst_obj = self.get_worst_archived_solution()
				
				# replace the worst solution (xG)
				if candidate.objective < worst_obj:
					self.archived_solutions[idx] = candidate
			else:
				# if better than the best objective, replace the closest solution
				_, best_obj = self.get_best_archived_solution()
				
				if candidate.objective < best_obj:
					self.archived_solutions[closest_idx] = candidate

				# otherwise, if better than the closest similar solution replace (xE)
				elif candidate.objective < self.archived_solutions[closest_idx].objective and closest_distance < self.Dsim:
					self.archived_solutions[closest_idx] = candidate


	def plot_on_contour(self):
		fig = plot_eggholder(2*512, ThreeD=False, Contour=True)
		ax = fig.axes[0]
		for sol in self.archived_solutions:
			ax.scatter(sol.coords[0], sol.coords[1])
		return fig

	def plot_on_surface(self):
		fig = plot_eggholder(2*512, ThreeD=True, Contour=False)
		ax = fig.axes[0]
		for sol in self.archived_solutions:
			ax.scatter(sol.coords[0], sol.coords[1], sol.objective)
		return fig	