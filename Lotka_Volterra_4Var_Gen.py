#--> Import standard python libraries
from math import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#--> Import some features of scipy to simulate the systems
#    or for matrix manipulation.
from scipy.integrate import odeint
from scipy.linalg import block_diag

#--> Import the PolynomialFeatures function from the sklearn
#    package to easily create the library of candidate functions
#    that will be used in the sparse regression problem.
from sklearn.preprocessing import PolynomialFeatures

#--> Import the sparse identification python package containing
#    the class to create sindy estimators.
import sparse_identification as sp
from sparse_identification.utils import derivative as spder
from sparse_identification.solvers import hard_threshold_lstsq_solve

def perturb(X, stdev, dt):
	if stdev == 0: return X, spder(X, dt)
	new_X = X + np.random.normal(0, stdev, np.shape(X))
	return new_X, spder(new_X, dt)

def Lotka_Volterra(x0, r, a, time, noise=0):

	"""
	This small function runs a simulation of the Lotka-Volterra system.
	Inputs
	------
	x0 : numpy array containing the initial condition.
	alpha, beta: Parameters of Lotka-Volterra system. 
				 alpha is 1-dimensional, and beta is 
				 a matrix. 
	time : numpy array for the evaluation of the state of
		   the Lotka-Volterra system at some given time instants.
	Outputs
	-------
	x : numpy two-dimensional array.
		State vector of the vector for the time instants
		specified in time.
	xdot : corresponding derivatives evaluated using
		   central differences.
	"""

	def dynamical_system(y,t):
		return (r + a @ y) * y

	x = odeint(dynamical_system,x0,time,mxstep=5000000)
	dt = time[1]-time[0]
	xdot = spder(x,dt)
	if noise == 0: return x, xdot

	return perturb(x, noise, dt)

def constraints(library):

	"""

	This function illustrates how to impose some
	user-defined constraints for the sparse identification.

	Input
	-----

	library : library object used for the sparse identification.

	Outputs
	-------

	C : two-dimensional numpy array.
		Constraints to be imposed on the regression coefficients.

	d : one-dimensional numpy array.
		Value of the constraints.

	"""

	#--> Recover the number of input and output features of the library.
	m = library.n_input_features_
	n = library.n_output_features_

	#--> Initialise the user-defined constraints matrix and vector.
	#    In this example, two different constraints are imposed.
	C = np.zeros((2, m*n))
	d = np.zeros((2,1))

	#--> Definition of the first constraint:
	#    In the x-equation, one imposes that xi[2] = -xi[1]
	#    Note: xi[0] corresponds to the bias, xi[1] to the coefficient
	#    for x(t) and xi[2] to the one for y(t).
	C[0, 1] = 1
	C[0, 2] = 1

	#--> Definition of the second constraint:
	#    In the y-equation, one imposes that xi[1] = 28
	#    Note: the n+ is because the coefficient xi[1] for
	#    the y-equation is the n+1th entry of the regression
	#    coefficients vector.
	C[1, n+1] = 1
	d[1] = 28

	return C, d

def Identified_Model(y, t, library, estimator) :

	'''
	Simulates the model from Sparse identification.

	Inputs
	------

	library: library object used in the sparse identification
			 (e.g. poly_lib = PolynomialFeatures(degree=3) )

	estimator: estimator object obtained from the sparse identification

	Output
	------

	dy : numpy array object containing the derivatives evaluated using the
		 model identified from sparse regression.

	'''

	dy = np.zeros_like(y)

	lib = library.fit_transform(y.reshape(1,-1))
	Theta = block_diag(lib, lib, lib, lib)
	dy = Theta.dot(estimator.coef_)

	return dy

def gen_init(r, a, t):
	initials = [[] for _ in range(10)]
	for i in range(10):
		total = 0
		while total < 5:
			x0 = np.random.uniform(0, 1, 4)
			x, dx = Lotka_Volterra(x0, r, a, t)
			if np.max(np.abs(x)) > 1 or np.any(np.isnan(x)) or np.all(x == 0):
				continue
			total += 1
			initials[i].append(x0)
	return initials

def traj(initials, r, a, t, noise, ret=False):
	lst = []
	i = 0
	while i < 10:
		x, dx = [], []
		for x0 in initials[i]:
			x_temp, dx_temp = Lotka_Volterra(x0, r, a, t, noise)
			x.append(x_temp)
			dx.append(dx_temp)
		
		x, dx = np.concatenate(x), np.concatenate(dx)
		library = PolynomialFeatures(degree=2, include_bias=True)
		Theta = library.fit_transform(x)
		n_lib = library.n_output_features_
		A = block_diag(Theta, Theta, Theta, Theta)
		b = dx.flatten(order='F')
		shols = sp.sindy(l1=0.1, solver='lstsq')
		try:
			shols.fit(A, b)
		except:
			continue
		x_ident = odeint(Identified_Model, x0, t, args=(library, shols))
		inf_norm = np.max(np.abs(x_ident.T - x_temp.T), axis=1)
		lst.append(inf_norm)
		i += 1

	if ret: return x_temp, x_ident
	return lst