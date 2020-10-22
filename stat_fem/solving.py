"""
Solving module just provides functional wrappers to the LinearSolver class for one-off computations.
For repeated use, please use the LinearSolver class
"""

from firedrake import COMM_SELF
from .LinearSolver import LinearSolver

def solve_posterior(A, x, b, G, data, params, ensemble_comm=COMM_SELF, scale_mean=False):
    """
    Solve for the FEM posterior conditioned on the data. solution is stored in the
    provided firedrake function x

    Note that the solution is only stored in the root of the ensemble comm if the
    forcing covariance solves are parallelized. The Firedrake function on other
    processes will not be modified.
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm)
    ls.set_params(params)
    ls.solve_posterior(x, scale_mean)

def solve_posterior_covariance(A, b, G, data, params, ensemble_comm=COMM_SELF, scale_mean=False):
    """
    solve for conditioned fem plus covariance in the data space

    returns solution as numpy arrays on the root process (rank 0)

    note that unlike the meshspace solver, this uses a return value rather than a
    Firedrake/PETSc style interface to create the solution. I was unable to get this
    to work by modifying the arrays in the function. This has the benefit of not requiring
    the user to pre-set the array sizes (the arrays are different sizes on the processes,
    as the solution is collected at the root of both the spatial comm and the ensemble comm)
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm)
    ls.set_params(params)
    return ls.solve_posterior_covariance(scale_mean)

def solve_prior_covariance(A, b, G, data, ensemble_comm=COMM_SELF):
    """
    solve base (prior) fem plus covariance interpolated to the data locations

    returns solution as numpy arrays on the root process (rank 0)

    note that unlike the meshspace solver, this uses a return value rather than a
    Firedrake/PETSc style interface to create the solution. I was unable to get this
    to work by modifying the arrays in the function. This has the benefit of not requiring
    the user to pre-set the array sizes (the arrays are different sizes on the processes,
    as the solution is collected at the root of both the spatial comm and the ensemble comm)

    Note that since the data locations are needed, this still requires an ObsData object.
    """

    return LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm).solve_prior()

def solve_prior_generating(A, b, G, data, params, ensemble_comm=COMM_SELF):
    "solve for the prior of the generating process"

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm)
    ls.set_params(params)
    return ls.solve_prior_generating()

def solve_posterior_generating(A, b, G, data, params, ensemble_comm=COMM_SELF):
    "solve for the posterior of the generating process"

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm)
    ls.set_params(params)
    return ls.solve_posterior_generating()

def predict_mean(A, b, G, data, params, coords, ensemble_comm=COMM_SELF, scale_mean=True):
    """
    predict mean data values at unmeasured locations

    returns vector of predicted sensor values on root process as numpy array. requires
    only a small overhead above the computational work of finding the posterior mean
    (i.e. get mean value at new sensor locations for "free" once you have solved the
    posterior)
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm)
    ls.set_params(params)
    return ls.predict_mean(coords, scale_mean)


def predict_covariance(A, b, G, data, params, coords, unc, ensemble_comm=COMM_SELF):
    """
    predict the mean and covariance of data values at unmeasured locations

    returns vector of predicted sensor values on root process as numpy array. requires
    doing an additional 2*n_pred FEM solves to get the full covariance at the new locations.
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm)
    ls.set_params(params)
    return ls.predict_covariance(coords, unc)
