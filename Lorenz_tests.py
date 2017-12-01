from Lorenz import *
import numpy as np

def simulate(t, x0, X, dX):
	poly_lib = PolynomialFeatures(degree=2, include_bias=True)
	lib = poly_lib.fit_transform(X)
	Theta = block_diag(lib, lib, lib)
	n_lib = poly_lib.n_output_features_
	b = dX.flatten(order='F')
	A = Theta
	C, d = constraints(poly_lib)
	estimator = sp.sindy(l1=0.01, solver='lasso')
	estimator.fit(A, b)
	Y  = odeint(Identified_Model, x0, t, args=(poly_lib, estimator))

	return Y

def main(vars1, vars2):
	(sigma1, rho1, beta1, x0) = vars1; (sigma2, rho2, beta2, xx0) = vars2 

	t = np.linspace(0,100,10000)
	X1, dX1 = Lorenz(x0, sigma1, rho1, beta1, t)
	Y1 = simulate(t, x0, X1, dX1)
	X2, dX2 = Lorenz(xx0, sigma2, rho2, beta2, t)
	Y2 = simulate(t, xx0, X2, dX2)

	#--> Plots the results to compare the dynamics of the identified system against the original one.
	plot_multi_results(t, X1, Y1, X2, Y2)
	plt.show()

main((10., 28., 8./3., np.array([-8., 7., 27.])), (10.1, 28., 8./3., np.array([-8., 7., 27.])))
main((10., 28., 8./3., np.array([-8., 7., 27.])), (10, 28.1, 8./3., np.array([-8., 7., 27.])))
main((10., 28., 8./3., np.array([-8., 7., 27.])), (10.1, 28., 8.5/3., np.array([-8., 7., 27.])))
main((10., 28., 8./3., np.array([-8., 7., 27.])), (10., 28., 8./3., np.array([-8.1, 7., 27.])))
main((10., 28., 8./3., np.array([-8., 7., 27.])), (10., 28., 8./3., np.array([-8., 7.1, 27.])))
main((10., 28., 8./3., np.array([-8., 7., 27.])), (10., 28., 8./3., np.array([-8., 7., 27.1])))
