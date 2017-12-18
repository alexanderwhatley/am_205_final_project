#######################################################################
#####                                                             #####
#####     SPARSE IDENTIFICATION OF NONLINEAR DYNAMICS (SINDy)     #####
#####     Application to the Lotka-Volterra system                #####
#####                                                             #####
#######################################################################

"""

This file provides a simple example and depiction of the SINDy algorithm
on a perturbed trajectory. The noise level used is 1e-3.

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

#--> Import helper functions in Lotka_Volterra_4Var_Gen.py
from Lotka_Volterra_4Var_Gen import *

def plot_results_multi(t, X1, Y1, X2, Y2, noise):

	"""

	Function to plot unperturbed and perturbed trajectories.

	Inputs
	------

	t : numpy array for the evaluation of the state of
		the Lotka-Volterra system at some given time instants.
	X1 : unperturbed true trajectory.
	Y1 : unperturbed identified trajectory.
	X2 : perturbed true trajectory.
	Y2 : perturbed identified trajectory.
	noise : float representing the standard deviation at which 
			the trajectory was perturbed.

	"""

	fig, ax = plt.subplots( 4 , 2 , sharex = True, figsize=(10,5) )
	plt.suptitle('Actual vs. Prediction - $t \in [0, {}], \sigma = {}$'.format(100, noise))

	def plot_side(j, X, Y):
		ax[0][j].plot(t  , X[:,0], color='b', label='Full simulation' )
		ax[0][j].plot(t , Y[:,0], color='r', linestyle='--', label='Identified model')
		ax[0][j].set_ylabel('x1(t)')
		ax[0][j].legend(loc='upper center', bbox_to_anchor=(.5, 1.33), ncol=2, frameon=False )
		ax[0][j].set_ylim(0, 1)

		ax[1][j].plot(t, X[:,1], color='b')
		ax[1][j].plot(t ,Y[:,1], color='r', ls='--')
		ax[1][j].set_ylabel('x2(t)')
		ax[1][j].set_ylim(0, 1)

		ax[2][j].plot(t, X[:,2], color='b')
		ax[2][j].plot(t ,Y[:,2], color='r', ls='--')
		ax[2][j].set_ylabel('x3(t)')
		ax[2][j].set_ylim(0, 1)

		ax[3][j].plot(t, X[:,3], color='b')
		ax[3][j].plot(t ,Y[:,3], color='r', ls='--')
		ax[3][j].set_ylabel('x4(t)')
		ax[3][j].set_xlabel('Time')
		ax[3][j].set_xlim(0, 100)
		ax[3][j].set_ylim(0, 1)

	plot_side(0, X1, Y1)
	plot_side(1, X2, Y2)
	plt.savefig('31c_error1.png')

	return

def plot_error(t, error1, error2, noise):

	"""

	Function to plot error between true and identified 
	trajectories for perturbed and unperturbed trajectories.

	Inputs
	------

	t : numpy array for the evaluation of the state of
		the Lotka-Volterra system at some given time instants.
	error1 : error between unperturbed true and identified trajectories.
	error2 : error between perturbed true and identified trajectories.
	noise : float representing the standard deviation at which 
			the trajectory was perturbed.

	"""

	fig, ax = plt.subplots( 4 , 2 , sharex = True, figsize=(10,5) )
	plt.suptitle('Prediction Error - $t \in [0, {}], \sigma = {}$'.format(100, noise))

	def plot_side(j, Z):
		ax[0][j].plot(t  , Z[:,0], color='b')
		ax[0][j].set_ylabel('x1(t) error')

		ax[1][j].plot(t, Z[:,1], color='b')
		ax[1][j].set_ylabel('x2(t) error')

		ax[2][j].plot(t, Z[:,2], color='b')
		ax[2][j].set_ylabel('x3(t) error')

		ax[3][j].plot(t, Z[:,3], color='b')
		ax[3][j].set_ylabel('x4(t) error')
		ax[3][j].set_xlabel('Time')
		ax[3][j].set_xlim(0, 100)

	plot_side(0, error1)
	plot_side(1, error2)
	plt.savefig('31c_error2.png')

	return

def main(noise):

	"""

	Function to plot error between true and identified 
	trajectories for perturbed and unperturbed trajectories.

	Inputs
	------

	t : numpy array for the evaluation of the state of
		the Lotka-Volterra system at some given time instants.
	error1 : error between unperturbed true and identified trajectories.
	error2 : error between perturbed true and identified trajectories.
	noise : float representing the standard deviation at which 
			the trajectory was perturbed.

	"""

	#--> Initializes variables and generates initial conditions.
	r = np.array([1, 0.72, 1.53, 1.27])
	a = np.array([[-1, -1.09, -1.52, 0], 
				  [0, -1*0.72, -0.44*0.72, -1.36*0.72], 
				  [-2.33*1.53, 0, -1.53, -0.47*1.53], 
				  [-1.21*1.27, -0.51*1.27, -0.35*1.27, -1.27]])
	t = np.linspace(0, 100, 2000)
	diff = []
	initials = gen_init(r, a, t)

	#--> Calculates identified trajectory for unperturbed and 
	#	 perturbed trajectories.
	X1, Y1 = model(initials, r, a, t, 0, error=True)
	X2, Y2 = model(initials, r, a, t, noise, error=True)

	#--> Calculates error between true and identified trajectories.
	error1 = np.absolute(Y1-X1)
	error2 = np.absolute(Y2-X2)

	#--> Plots the error and trajectories obtained above.
	plot_error(t, error1, error2, noise)
	plot_results_multi(t, X1, Y1, X2, Y2, noise)
	plt.show()

main(1e-3)