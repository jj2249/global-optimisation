import numpy as np
import matplotlib.pyplot as plt


def eggholder_naive(x):
	"""
	Naive implementation of eggholder function. Maps n-dimensional space to a scalar, n>=2

	Inputs:
	- x: length n vector of input coordinates

	Outputs:
	- __: scalar output of function
	"""

	assert type(x) == np.ndarray, "Control vector must be a numpy array"
	assert x.shape[0] >= 2, "Eggholder function not defined on spaces below 2D"
	# dimension of problem
	n = x.shape[0]
	# sum container
	total = 0
	# implement sum over each pair of adjacent dimensions
	for i in range(n-1):
		# store current x vals
		xi = x[i]
		xip1 = x[i+1]

		t1 = -(xip1+47)*np.sin(np.sqrt(np.abs(xip1+0.5*xi+47)))
		t2 = -xi*np.sin(np.sqrt(np.abs(xi-xip1-47)))

		total = total + t1 + t2

	return total


def eggholder(x):
	"""
	Vectorised implementation of eggholder function. Maps n-dimensional space to a scalar, n>=2

	Inputs:
	- x: length n vector of input coordinates

	Outputs:
	- __: scalar output of function
	"""
	assert type(x) == np.ndarray, "Control vector must be a numpy array"
	assert x.shape[0] >= 2, "Eggholder function not defined on spaces below 2D"
	# dimension of problem
	n = x.shape[0]

	# roll x to align the adjacent dimensions
	xr = np.roll(x, -1, axis=0)[:n-1]
	# dont need the final pair (x0, xn-1)
	xt = x[:n-1]

	# each term in the sum
	x1 = -xr-47
	x2 = np.sqrt(np.abs(xr+0.5*xt+47))
	x3 = -xt
	x4 = np.sqrt(np.abs(xt-xr-47))

	# summand for each pair
	summand = x1*np.sin(x2) + x3*np.sin(x4)
	# return sum
	return np.sum(summand, axis=0)


def plot_eggholder(samps, ThreeD=True, Contour=False):
	# axes
	x1 = np.linspace(-512, 512, samps)
	x2 = np.linspace(-512, 512, samps)
	# mesh
	xx1, xx2 = np.meshgrid(x1, x2)
	x = np.array([xx1, xx2])
	# output surface
	z = eggholder(x)

	fig = plt.figure()
	# conditions for choice of plots
	if ThreeD and Contour:
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.plot_surface(xx1, xx2, z, cmap='viridis', linewidth=0, antialiased=False)
		ax2 = fig.add_subplot(122)
		ax2.contour(xx1, xx2, z, levels=10)

	if Contour and (not ThreeD):
		ax = fig.add_subplot()
		ax.contour(xx1, xx2, z, levels=10)

	if (not Contour) and ThreeD:
		ax = fig.add_subplot(projection='3d')
		ax.plot_surface(xx1, xx2, z, cmap='viridis', linewidth=0, antialiased=False)

	return fig


def arctan(num, den):
	if den == 0:
		return np.pi/2.
	else:
		return np.arctan(num/den)