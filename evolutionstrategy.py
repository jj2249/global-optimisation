import numpy as np
from functions import *
from solution import esSolution
import matplotlib.pyplot as plt
from archive import Archive

class EvolutionStrategy:
    def __init__(self, dimension, nparents, noffspring, epsilon, seed):
        self.dimension = dimension
        
        # parameters for the overall strategy
        self.nparents = nparents
        self.noffspring = noffspring
        self.epsilon = epsilon

        # model not initially converged
        self.converged = False

        # Generate an initial population
        self.offspring = None
        self.generate_offspring()
        self.generation = 0
        self.parents = None

        self.archive = Archive(30, 10., 0.1)

        # seed the rng
        self.seed = seed
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
        - 1 objective evaluation per offspring
        """
        offspring = []
        for _ in range(self.noffspring):
            # x = 500*np.random.randn(self.dimension)
            # reject until in the feasible region
            # while not (np.all(x <=512.) and np.all(x >= -512)):
                # x = 500*np.random.randn(self.dimension)
            x = np.random.uniform(low=-512., high=512., size=self.dimension)

            # new offspring at the proposed coordinates with Identitiy covariance matrix
            offspring.append(esSolution(x, 2500*np.ones(self.dimension), np.zeros(int(self.dimension*(self.dimension-1)/2.))))

        # full initial population
        self.offspring = np.array(offspring)


    def survival(self, record_parents=False, folder=None, fig=None, title=''):
        """
        Select the best solutions (lowest objective) and assign as parents
        - no objective evaluations
        """
        # collect offspring objectives for sorting
        objectivef = np.array([off.objective for off in self.offspring])

        # sort the offspring in increasing objective order (best to worst)
        idx = np.argsort(objectivef)

        # assign the the new parents
        strongest = np.take(self.offspring, idx, axis=0)
        self.parents = strongest[:self.nparents]
        for parent in self.parents:
            self.archive.check_candidate(parent)

        self.generation += 1

        if record_parents:
            fig = self.plot_parents_on_contour(title="Generation {}".format(self.generation))
            plt.savefig(folder+"/file%02d"%(self.generation))
            plt.close(fig)


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
                # unbiased coin toss to determine which parent passes it's coordinate on
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

                # new child initialised with infinite objective to avoid calculating the objective before we know that this configuration is positive definite
                # can't use None type since this will cause constructor to evaluate the objective
                new_child = esSolution(child, variances=new_variances, rotangles=new_rotangles, objective=np.inf)
                try:
                    C = new_child.covariance_matrix()
                    Cchol = np.linalg.cholesky(C+CHOL_FAC*np.eye(self.dimension))
                    is_pos_def = True
                except np.linalg.LinAlgError:
                    pass

            # evaluate the valid child's objective and add to the container of offspring
            new_child.evaluate_objective()
            new_offspring.append(new_child)

        # store the new set of offspring
        self.offspring = np.array(new_offspring)


    def global_recombine(self):
        """
        Generate new offspring using GLOBAL recombination
        """
        new_offspring = []

        for _ in range(self.noffspring):
            # child coordinates determined by discrete recombination
            child = np.zeros(self.dimension)
            for i in range(self.dimension):
                # roulette wheel to select parents randomly FOR EACH COORDINATE
                p1c = np.random.randint(0, self.nparents)
                child[i] = self.parents[p1c].coords[i]

            # need to ensure positive definiteness of covariance matrix
            is_pos_def = False
            while not is_pos_def:
                # resample parents until a valid configuration found
                new_variances = []
                new_rotangles = []
                weight = 0.5
                for i in range(self.dimension):
                    p1s = np.random.randint(0, self.nparents)
                    p2s = np.random.randint(0, self.nparents)

                    # new strategy parameters are the weighted sum of the two parents
                    new_variances.append(weight*self.parents[p1s].variances[i] + (1-weight)*self.parents[p2s].variances[i])

                for i in range(int(self.dimension*(self.dimension-1)/2)):
                    p1s = np.random.randint(0, self.nparents)
                    p2s = np.random.randint(0, self.nparents)

                    new_rotangles.append(weight*self.parents[p1s].rotangles[i] + (1-weight)*self.parents[p2s].rotangles[i])

                # new child initialised with infinite objective to avoid calculating the objective before we know that this configuration is positive definite
                # can't use None type since this will cause constructor to evaluate the objective
                new_child = esSolution(child, variances=np.array(new_variances), rotangles=np.array(new_rotangles), objective=np.inf)
                try:
                    C = new_child.covariance_matrix()
                    Cchol = np.linalg.cholesky(C+CHOL_FAC*np.eye(self.dimension))
                    is_pos_def = True
                except np.linalg.LinAlgError:
                    pass

            # evaluate the valid child's objective and add to the container of offspring
            new_child.evaluate_objective()
            new_offspring.append(new_child)

        # store the new set of offspring
        self.offspring = np.array(new_offspring)
    

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
        # sorted objectives of the parents
        objectives = np.sort(np.array([parent.objective for parent in self.parents]))
        
        # absolute difference between best and worst parent
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
        objectives = np.sort(np.array([parent.objective for parent in self.parents]))
        return objectives[0]


    def mean_objective(self):
        """
        Return the mean objective value of parents
        """
        objectives = np.array([parent.objective for parent in self.parents])
        return np.mean(objectives, axis=0)


    def plot_parents_on_contour(self, title=''):
        """
        Plot the offspring on the contour map
        """
        fig = plot_eggholder(2*512, ThreeD=False, Contour=True, title=title)
        ax = fig.axes[0]
        for parent in self.parents:
            ax.scatter(parent.coords[0], parent.coords[1], color='red')
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