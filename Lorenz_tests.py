from Lorenz import *
import numpy as np

def main(sigma, rho, beta, x0):
	t = np.linspace(0,100,10000)

	#--> Run the Lorenz system to produce the data to be used in the sparse identification.
	X, dX = Lorenz(x0, sigma, rho, beta, t)

	#-----------------------------------------------------------------------
	#-----                                                             -----
	#-----     Sparse Identification of Nonlinear Dynamics (SINDY)     -----
	#-----                                                             -----
	#-----------------------------------------------------------------------

	# ---> Compute the Library of polynomial features.
	poly_lib = PolynomialFeatures(degree=2, include_bias=True)
	lib = poly_lib.fit_transform(X)
	Theta = block_diag(lib, lib, lib)
	n_lib = poly_lib.n_output_features_

	b = dX.flatten(order='F')
	A = Theta

	# ---> Specify the user-defined constraints.
	C, d = constraints(poly_lib)

	# ---> Create a linear regression estimator using sindy and identify the underlying equations.
	estimator = sp.sindy(l1=0.01, solver='lasso')
	estimator.fit(A, b)#, eq=[C, d])

	#--> Simulates the identified model.
	Y  = odeint(Identified_Model, x0, t, args=(poly_lib, estimator))

	#--> Plots the results to compare the dynamics of the identified system against the original one.
	plot_results(t, X, Y)
	plt.show()

main(10., 28., 8./3., np.array([-8., 7., 27.]))
main(10.1, 28., 8./3., np.array([-8., 7., 27.]))
main(10.5, 28., 8./3., np.array([-8., 7., 27.]))
main(10., 28., 8./3., np.array([-8.1, 7., 27.]))
main(10., 28., 8./3., np.array([-8., 7.1, 27.]))
main(10., 28., 8./3., np.array([-8., 7., 27.1]))
