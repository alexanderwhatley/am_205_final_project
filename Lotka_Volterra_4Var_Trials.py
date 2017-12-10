import Lotka_Volterra_4Var as lv4 
import numpy as np 

num_trials = 1000

r = np.array([1, 0.72, 1.53, 1.27])
a = np.array([[1, 1.09, 1.52, 0], 
			  [0, 1, 0.44, 1.36], 
			  [2.33, 0, 1, 0.47], 
			  [1.21, 0.51, 0.35, 1]])

t = np.linspace(0,500,5000)
x0 = np.random.rand(4)
X1, dX1 = lv4.Lotka_Volterra(x0, r, a, t)
Y1 = lv4.SINDy(x0, t, X1, dX1)
error_X = np.zeros(num_trials)
error_Y = np.zeros(num_trials)

for i in range(num_trials):
	X2, dX2 = lv4.perturb(X1, 0.001)
	Y2 = lv4.SINDy(x0, t, X2, dX2)
	error_X[i] = np.average(X2-X1)
	error_Y[i] = np.average(Y2-Y1)

ratio = error_X/error_Y
print("Minimum ratio: {}".format(np.min(ratio)))
print("Maximum ratio: {}".format(np.max(ratio)))
print("Average ratio: {}".format(np.average(ratio)))
print("Minimum error Y: {}".format(np.min(error_Y)))
print("Maximum error Y: {}".format(np.max(error_Y)))
print("Average error Y: {}".format(np.average(error_Y)))


