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
    if noise == 0: return x, xdot

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

def SINDy(x0, t, X, dX):
    # ---> Compute the Library of polynomial features.
    poly_lib = PolynomialFeatures(degree=2, include_bias=True)
    lib = poly_lib.fit_transform(X)
    Theta = block_diag(lib, lib, lib, lib)
    n_lib = poly_lib.n_output_features_

    b = dX.flatten(order='F')
    A = Theta

    # ---> Specify the user-defined constraints.
    C, d = constraints(poly_lib)

    # ---> Create a linear regression estimator using sindy and identify the underlying equations.
    estimator = sp.sindy(l1=0.01, solver='lstsq')
    estimator.fit(A, b)#, eq=[C, d])
    coeffs = hard_threshold_lstsq_solve(A, b)
    print(coeffs)

    #--> Simulates the identified model.
    Y  = odeint(Identified_Model, x0, t, args=(poly_lib, estimator),mxstep=5000000)

    return Y

def plot_results_multi(t, X1, Y1, X2, Y2):

    """

    Function to plot the results. No need to comment.

    """

    fig, ax = plt.subplots( 4 , 2 , sharex = True, figsize=(20,5) )

    def plot_side(j, X, Y):
        ax[0][j].plot(t  , X[:,0], color='b', label='Full simulation' )
        ax[0][j].plot(t , Y[:,0], color='r', linestyle='--', label='Identified model')
        ax[0][j].set_ylabel('x1(t)')
        ax[0][j].legend(loc='upper center', bbox_to_anchor=(.5, 1.33), ncol=2, frameon=False )

        ax[1][j].plot(t, X[:,1], color='b')
        ax[1][j].plot(t ,Y[:,1], color='r', ls='--')
        ax[1][j].set_ylabel('x2(t)')

        ax[2][j].plot(t, X[:,2], color='b')
        ax[2][j].plot(t ,Y[:,2], color='r', ls='--')
        ax[2][j].set_ylabel('x3(t)')

        ax[3][j].plot(t, X[:,3], color='b')
        ax[3][j].plot(t ,Y[:,3], color='r', ls='--')
        ax[3][j].set_ylabel('x4(t)')
        ax[3][j].set_xlabel('Time')
        ax[3][j].set_xlim(0, 200)

    plot_side(0, X1, Y1)
    plot_side(1, X2, Y2)

    return

def plot_error(t, X1, Y1, X2, Y2):
    error1 = np.absolute(Y1-X1)
    error2 = np.absolute(Y2-X2)

    fig, ax = plt.subplots( 4 , 2 , sharex = True, figsize=(20,5) )

    def plot_side(j, Z):
        ax[0][j].plot(t  , Z[:,0], color='b')
        ax[0][j].set_ylabel('x1(t) error')

        ax[1][j].plot(t, Z[:,1], color='b')
        ax[1][j].set_ylabel('x2(t) error')

        ax[2][j].plot(t, Z[:,2], color='b')
        ax[2][j].set_ylabel('x3(t)')

        ax[3][j].plot(t, Z[:,3], color='b')
        ax[3][j].set_ylabel('x4(t) error')
        ax[3][j].set_xlabel('Time')
        ax[3][j].set_xlim(0, 200)

    plot_side(0, error1)
    plot_side(1, error2)

    return

if __name__ == '__main__':

    #--> Sets the parameters for the Lotka-Volterra system.
    r = np.array([1, 0.72, 1.53, 1.27])
    a = np.array([[1, 1.09, 1.52, 0], 
                  [0, 1, 0.44, 1.36], 
                  [2.33, 0, 1, 0.47], 
                  [1.21, 0.51, 0.35, 1]])

    t = np.linspace(0,200,50000)

    #--> Run the Lotka-Volterra system to produce the data to be used in the sparse identification.
    x0 = np.random.rand(4)
    X1, dX1 = Lotka_Volterra(x0, r, a, t)
    print("cat")
    X2, dX2 = perturb(X1, 0.01)

    Y1 = SINDy(x0, t, X1, dX1)
    Y2 = SINDy(x0, t, X1, dX1)

    #--> Plots the results to compare the dynamics of the identified system against the original one.
    plot_error(t, X1, Y1, X2, Y2)
    plot_results_multi(t, X1, Y1, X2, Y2)

    plt.show()