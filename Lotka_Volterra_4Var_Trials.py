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
from sparse_identification.utils import derivative

#--> Import helper functions in Lotka_Volterra_4Var_Gen.py
from Lotka_Volterra_4Var_Gen import *

r = np.array([1, 0.72, 1.53, 1.27])
a = np.array([[-1, -1.09, -1.52, 0], 
			  [0, -1*0.72, -0.44*0.72, -1.36*0.72], 
			  [-2.33*1.53, 0, -1.53, -0.47*1.53], 
			  [-1.21*1.27, -0.51*1.27, -0.35*1.27, -1.27]])
t = np.linspace(0, 100, 2000)
noise_level = np.logspace(-5, -1, 50)
diff = []
initials = gen_init(r, a, t)

for noise in noise_level:
	print(noise)
	diff.append(np.mean(traj(initials, r, a, t, noise), axis=0))

plt.plot(np.log10(noise_level), np.log10(diff))
plt.title('Infinity Norm between Trajectories - $t \in [0, 100]$')
plt.xlabel('Log(Noise Level)')
plt.ylabel('Log(Infinity Norm)')
plt.savefig('31c_trials1.png'.format())
plt.show()