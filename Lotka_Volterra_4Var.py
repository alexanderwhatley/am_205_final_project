# Method that uses derivative approximation 
# Does not appear to work very well 
from cvxopt import solvers
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from scipy.integrate import odeint
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from sklearn.preprocessing import PolynomialFeatures
from Lotka_Volterra import Lotka_Volterra
from sparse_identification.solvers import hard_threshold_lstsq_solve
from sparse_identification.utils import derivative

def approx_derivative(x, t):
    # compute differences in x
    dx = np.zeros_like(x)
    dx[1:-1, :] = (x[2:, :] - x[:-2, :])
    dx[0,:]    = (x[1,:] - x[0,:])
    dx[-1,:]   = (x[-1,:] - x[-2,:])
    # compute differences in t
    dt = np.zeros_like(t)
    dt[1:-1] = (t[2:] - t[:-2])
    dt[0]    = (t[1] - t[0])
    dt[-1]   = (t[-1] - t[-2])
    
    return (dx/dt[:, None]).flatten(order='F')

def Lotka_Volterra(x0, r, a, time):
    def dynamical_system(y,t):

        dy = np.zeros_like(y)
        for i in range(4):
            dy[i] = r[i]*y[i]*(1-a[i][0]*y[0]-a[i][1]*y[1]-a[i][2]*y[2]-a[i][3]*y[3])

        return dy

    x = odeint(dynamical_system,x0,time,mxstep=5000000)
    dt = time[1]-time[0]
    xdot = derivative(x,dt)

    return x, xdot

def perturb(X, stdev):
    def gen():
        new_X = np.zeros(np.shape(X))
        for i in range(len(new_X)):
            (d1, d2, d3, d4) = np.random.normal(0, stdev, 4)
            new_X[i] = [(X[i][0]+d1), (X[i][1]+d2), (X[i][2]+d3), (X[i][3]+d4)]
        return new_X

    while True:
        new_X = gen()
        print(new_X)
        if np.max(new_X) != 1: return new_X, derivative(X)

r = np.array([1, 0.72, 1.53, 1.27])
a = np.array([[1, 1.09, 1.52, 0], 
              [0, 1, 0.44, 1.36], 
              [2.33, 0, 1, 0.47], 
              [1.21, 0.51, 0.35, 1]])

t = np.linspace(0,200,50000)

trajectories = []
total = 0
while total < 5:
    x0 = np.random.rand(4) # Initial condition.
    x, dx = perturb(Lotka_Volterra(x0, r, a, t)[0], 0.01)
    if np.max(np.abs(x)) > 1:
        continue
    trajectories.append(x)
    total += 1

trajectories = np.array(trajectories)
l1 = 0.01
degree = 2
num_coeffs = 10
library = PolynomialFeatures(degree=degree, include_bias=True)

x0 = [1e-3] * (3*num_coeffs) # initialize coefficients to be small values 
l2 = 0.01
iter_num = 0

def minimize_diff(x0):
    global iter_num
    diff_sq = 0
    A, b = [], []
    for ind, trajectory in enumerate(trajectories):
        deriv_est_lhs = approx_derivative(trajectory, t) # approximated derivates in LHS of system 
        b.append(deriv_est_lhs)
        Theta = library.fit_transform(trajectory)
        A_traj = block_diag(Theta, Theta, Theta, Theta)
        A.append(A_traj)
    A = np.concatenate(A)
    b = np.concatenate(b)
    coeffs = hard_threshold_lstsq_solve(A, b)
    #coeffs = least_squares(lambda x: A @ x - b, x0).x
    return coeffs

for _ in range(5):
    coeffs = minimize_diff(x0)
    for i in range(4):
        coeff_block = coeffs[i*num_coeffs:(i+1)*num_coeffs]
        xmax = abs(coeff_block[np.nonzero(coeff_block)]).mean()
        to_remove = [k for k in range(len(coeff_block)) if abs(coeff_block[k]) < l1*xmax]
        for k in to_remove:
            coeffs[i*num_coeffs + k] = 0
    x0 = coeffs
    print('After iteration {} we have the coefficients'.format(_))
    print(x0)