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

#--> Defines various functions used in this script.

def Lotka_Volterra(x0, r, a, time):

    """

    This small function runs a simulation of the Lotka-Volterra system.

    Inputs
    ------

    x0 : numpy array containing the initial condition.

    r, a : parameters of the Lotka-Volterra
     system.

    time : numpy array for the evaluation of the state of
           the Lotka-Volterra
            system at some given time instants.

    Outputs
    -------

    x : numpy two-dimensional array.
        State vector of the vector for the time instants
        specified in time.

    xdot : corresponding derivatives evaluated using
           central differences.

    """

    def dynamical_system(y,t):

        dy = np.zeros_like(y)
        for i in range(4):
            dy[i] = r[i]*y[i]*(1-a[i][0]*y[0]-a[i][1]*y[1]-a[i][2]*y[2]-a[i][3]*y[3])

        return dy

    x = odeint(dynamical_system,x0,time,mxstep=5000000)
    dt = time[1]-time[0]
    xdot = spder(x,dt)

    return x, xdot

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

#-----------------------------------------------------------------------
#-----                                                             -----
#-----     Sparse Identification of Nonlinear Dynamics (SINDY)     -----
#-----                                                             -----
#-----------------------------------------------------------------------

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
    estimator = sp.sindy(l1=0.01, solver='lasso')
    estimator.fit(A, b)#, eq=[C, d])

    #--> Simulates the identified model.
    Y  = odeint(Identified_Model, x0, t, args=(poly_lib, estimator),mxstep=5000000)

    return Y

def plot_results(t, X, Y):

    """

    Function to plot the results. No need to comment.

    """

    fig, ax = plt.subplots( 4 , 1 , sharex = True, figsize=(10,5) )

    ax[0].plot(t  , X[:,0], label='Full simulation' )
    ax[0].plot(t , Y[:,0], linestyle='--', label='Identified model')
    ax[0].set_ylabel('x1(t)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(.5, 1.33), ncol=2, frameon=False )

    ax[1].plot(t, X[:,1])
    ax[1].plot(t ,Y[:,1], ls='--')
    ax[1].set_ylabel('x2(t)')

    ax[2].plot(t, X[:,2])
    ax[2].plot(t ,Y[:,2], ls='--')
    ax[2].set_ylabel('x3(t)')

    ax[3].plot(t, X[:,3])
    ax[3].plot(t ,Y[:,3], ls='--')
    ax[3].set_ylabel('x4(t)')
    ax[3].set_xlabel('Time')
    ax[3].set_xlim(0, 200)

    return

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

def perturb(X, stdev):
    new_X = np.zeros(np.shape(X))
    for i in range(len(new_X)):
        (d1, d2, d3, d4) = np.random.normal(0, stdev, 4)
        new_X[i] = [(X[i][0]+d1), (X[i][1]+d2), (X[i][2]+d3), (X[i][3]+d4)]
    return new_X, spder(new_X)

if __name__ == '__main__':

    #--> Sets the parameters for the Lotka-Volterra system.
    r = np.array([1, 0.72, 1.53, 1.27])
    a = np.array([[1, 1.09, 1.52, 0], 
                  [0, 1, 0.44, 1.36], 
                  [2.33, 0, 1, 0.47], 
                  [1.21, 0.51, 0.35, 1]])

    t = np.linspace(0,500,5000)

    #--> Run the Lotka-Volterra system to produce the data to be used in the sparse identification.
    x0 = np.random.rand(4)
    x0 = np.array([0.25, 0.75, 0.6, 0.8])
    X1, dX1 = Lotka_Volterra(x0, r, a, t)
    X2, dX2 = perturb(X1, 0.05)

    Y1 = SINDy(x0, t, X1, dX1)
    Y2 = SINDy(x0, t, X2, dX2)
    error_X = np.average(X2-X1)
    error_Y = np.average(Y2-Y1)
    print(error_X)
    print(error_Y)
    print(error_Y/error_X)

    #--> Plots the results to compare the dynamics of the identified system against the original one.
    #plot_results_multi(t, X1, Y1, X2, Y2)
    plot_results(t, X1, Y1)
    plot_results(t, X2, Y2)
    plt.show()
