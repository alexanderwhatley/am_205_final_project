#######################################################################
#####                                                             #####
#####     SPARSE IDENTIFICATION OF NONLINEAR DYNAMICS (SINDy)     #####
#####     Application to the Lotka-Volterra system                #####
#####                                                             #####
#######################################################################

"""

This small example illustrates the identification of a nonlinear
dynamical system using the data-driven approach SINDy with constraints
by Loiseau & Brunton (submitted to JFM Rapids).

Note: The sklearn python package is required for this example.
----

Contact: loiseau@mech.kth.se

"""


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

#--> Defines various functions used in this script.

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

# from sklearn.preprocessing import PolynomialFeatures
def make_coefficients(r, a, num_terms):
	num_vars = 4
	coeffs = np.zeros((num_vars, num_terms))
	for i in range(num_vars):
		coeffs[i, i+1] = r[i]
	coeffs[0, [5, 6, 7, 8]] = a[0]
	coeffs[1, [6, 9, 10, 11]] = a[1]
	coeffs[2, [7, 10, 12, 13]] = a[2]
	coeffs[3, [8, 11, 13, 14]] = a[3]
	
	return coeffs.ravel()

r = np.array([1, 0.72, 1.53, 1.27])
a = np.array([[-1, -1.09, -1.52, 0], 
			  [0, -1*0.72, -0.44*0.72, -1.36*0.72], 
			  [-2.33*1.53, 0, -1.53, -0.47*1.53], 
			  [-1.21*1.27, -0.51*1.27, -0.35*1.27, -1.27]])
true_coeffs = make_coefficients(r, a, 15)
t = np.linspace(0, 100, 2000)
noise_level = np.logspace(-5, -1, 50)
dist, sparsity = [], []

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

for noise in noise_level:
	print(noise)
	i = 0
	noise_dist, noise_sparsity = [], []
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
		coefs = shols.coef_
		noise_dist.append(np.linalg.norm(coefs - true_coeffs, ord=2))
		num_mismatch = np.sum([coefs[i] != 0 and true_coeffs[i] == 0 
							   for i in range(len(coefs))])
		noise_sparsity.append(num_mismatch)
		i += 1
	dist.append(np.mean(noise_dist))
	sparsity.append(np.mean(noise_sparsity))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(np.log10(noise_level), dist)
ax[0].set_title('Distance Between Coefficients')
ax[0].set_xlabel('Log(Noise Level)')
ax[0].set_ylabel('Distance')
ax[1].plot(np.log10(noise_level), sparsity)
ax[1].set_title('Number of Mismatched Coefficients')
ax[1].set_xlabel('Log(Noise Level)')
ax[1].set_ylabel('# Mismatched Coefficients')
plt.savefig('31c_coeffs.png')
plt.show()
