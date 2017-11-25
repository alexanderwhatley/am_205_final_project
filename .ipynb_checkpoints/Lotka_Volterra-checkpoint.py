import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.linalg import block_diag
from sklearn.preprocessing import PolynomialFeatures
import sparse_identification as sp

def Lotka_Volterra(x0, alpha, beta, time):

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
        dy = (alpha + beta @ y) * y
        return dy

    x = odeint(dynamical_system,x0,time)
    dt = time[1]-time[0]
    from sparse_identification.utils import derivative
    xdot = derivative(x,dt)
    
    return x, xdot