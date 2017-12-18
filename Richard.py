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

def Lotka_Volterra(x0, r, a, time):
    def dynamical_system(y,t):
        dy = np.zeros_like(y)
        for i in range(4):
            dy[i] = r[i]*y[i]*(1-a[i][0]*y[0]-a[i][1]*y[1]-a[i][2]*y[2]-a[i][3]*y[3])
        return dy

    x = odeint(dynamical_system, x0, time, mxstep=5000000)
    dt = time[1] - time[0]
    xdot = spder(x, dt)

    return x, xdot

def Identified_Model(y, t, library, estimator):

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
        x_temp, dx_temp = Lotka_Volterra(x0, r, a, t)
        if np.max(np.abs(x_temp)) > 1 or np.any(np.isnan(x_temp)) or np.all(x_temp == 0):
            continue
        total += 1
        print(total)
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

