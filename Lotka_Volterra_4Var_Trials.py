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

#--> Defines various functions used in this script.

def perturb(X, stdev):
    if stdev == 0: return X, spder(X)
    new_X = X + np.random.normal(0, stdev, np.shape(X))
    return new_X, spder(new_X)

def Lotka_Volterra(x0, r, a, time, noise=0):
    def dynamical_system(y,t):
        dy = np.zeros_like(y)
        for i in range(4):
            dy[i] = r[i]*y[i]*(1-a[i][0]*y[0]-a[i][1]*y[1]-a[i][2]*y[2]-a[i][3]*y[3])
        return dy

    x = odeint(dynamical_system,x0,time,mxstep=5000000)
    dt = time[1]-time[0]
    xdot = spder(x,dt)

    return perturb(x, noise)

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

def main(time_range):

    #--> Sets the parameters for the Lotka-Volterra system.
    r = np.array([1, 0.72, 1.53, 1.27])
    a = np.array([[1, 1.09, 1.52, 0], 
                [0, 1, 0.44, 1.36], 
                [2.33, 0, 1, 0.47], 
                [1.21, 0.51, 0.35, 1]])

    t = np.linspace(0,time_range,50000)

    noise_level = np.logspace(-5, -1, 50)
    diff = []
    for noise in noise_level:
        print(noise)
        noise_diff = []
        for _ in range(5):
            total = 0
            x0 = np.random.rand(4) # Initial condition.
            x, dx = Lotka_Volterra(x0, r, a, t, noise=noise)

            # ---> Compute the Library of polynomial features.
            poly_lib = PolynomialFeatures(degree=2, include_bias=True)
            lib = poly_lib.fit_transform(x)
            Theta = block_diag(lib, lib, lib, lib)
            n_lib = poly_lib.n_output_features_

            b = dx.flatten(order='F')
            A = Theta

            # ---> Specify the user-defined constraints.
            C, d = constraints(poly_lib)

            # ---> Create a linear regression estimator using sindy and identify the underlying equations.
            estimator = sp.sindy(l1=0.01, solver='lstsq')
            estimator.fit(A, b)#, eq=[C, d])
            coeffs = hard_threshold_lstsq_solve(A, b)

            #--> Simulates the identified model.
            Y  = odeint(Identified_Model, x0, t, args=(poly_lib, estimator),mxstep=5000000)

            inf_norm = np.max(np.abs(Y.T - x.T), axis=1)
            noise_diff.append(inf_norm)
        diff.append(np.mean(noise_diff, axis=0))


    plt.plot(np.log10(noise_level), np.log10(diff))
    plt.title('Infinity Norm between Trajectories - $t \in [0, {}]$'.format(time_range))
    plt.xlabel('Log(Noise Level)')
    plt.ylabel('Log(Infinity Norm)')
    plt.savefig('31_c{}'.format(time_range))
    plt.show()

main(5)
main(20)
