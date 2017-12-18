#######################################################################
#####                                                             #####
#####     SPARSE IDENTIFICATION OF NONLINEAR DYNAMICS (SINDy)     #####
#####     Application to the Lotka-Volterra system                #####
#####                                                             #####
#######################################################################

"""

This file calculates the infinity norm between trajectories as a 
function of the number of time steps used for t \in [0, 100], from
1000 to 5000.

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
from sparse_identification.utils import derivative

#--> Import helper functions in Lotka_Volterra_4Var_Gen.py
from Lotka_Volterra_4Var_Gen import *

#--> Initializes variables and generates initial conditions.
r = np.array([1, 0.72, 1.53, 1.27])
a = np.array([[-1, -1.09, -1.52, 0], 
			  [0, -1*0.72, -0.44*0.72, -1.36*0.72], 
			  [-2.33*1.53, 0, -1.53, -0.47*1.53], 
			  [-1.21*1.27, -0.51*1.27, -0.35*1.27, -1.27]])
noise = 1e-5
time_steps = np.linspace(1000, 5000, 26)
diff = []
initials = gen_init(r, a, np.linspace(0, 100, 2000))

#--> Computes average infinity norm between trajectories for 
#	 each number of time steps.
for step in time_steps:
	t = np.linspace(0, 100, step)
	print(step)
	diff.append(np.mean(model(initials, r, a, t, noise), axis=0))

#--> Plots average infinity norm as a function of number of time steps.
plt.plot(time_steps, np.log10(diff))
plt.title('Infinity Norm between Trajectories - $t \in [0, 100]$')
plt.xlabel('Time Steps')
plt.ylabel('Log(Infinity Norm)')
plt.savefig('31c_steps1.png')
plt.show()