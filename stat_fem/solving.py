import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import LinAlgError
from firedrake import COMM_WORLD, COMM_SELF
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.matrix import Matrix
from firedrake.vector import Vector
from firedrake.solving import solve
from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix
from .ObsData import ObsData
from .solving_utils import _solve_forcing_covariance

def solve_posterior(A, x, b, G, data, params, ensemble_comm=COMM_SELF):
    """
    Solve for the FEM posterior conditioned on the data. solution is stored in the
    provided firedrake function x

    Note that the solution is only stored in the root of the ensemble comm if the
    forcing covariance solves are parallelized. The Firedrake function on other
    processes will not be modified.
    """

    if not isinstance(A, Matrix):
       raise TypeError("A must be a firedrake matrix")
    if not isinstance(x, (Function, Vector)):
        raise TypeError("x must be a firedrake function or vector")
    if not isinstance(b, (Function, Vector)):
        raise TypeError("b must be a firedrake function or vector")
    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a forcing covariance")
    if not isinstance(data, ObsData):
        raise TypeError("data must be an ObsData type")
    if not isinstance(ensemble_comm, type(COMM_WORLD)):
        raise TypeError("ensemble_comm must be an MPI communicator created with a firedrake Ensemble")

    params = np.array(params, dtype=np.float64)
    assert params.shape == (3,), "bad shape for model discrepancy parameters"
    rho = np.exp(params[0])

    # create interpolation matrix

    im = InterpolationMatrix(G.function_space, data.get_coords())

    # all processes participate in solving for the interpolated forcing covariance
    # returns a numpy array to root and dummy arrays to others

    Cu = im.interp_covariance_to_data(G, A, ensemble_comm)

    # remaining solves are just done on ensemble root

    if ensemble_comm.rank == 0:

        solve(A, x, b)

        if G.comm.rank == 0:
            Ks = data.calc_K_plus_sigma(params[1:])
            try:
                LK = cho_factor(Ks)
            except LinAlgError:
                raise LinAlgError("Error attempting to compute the Cholesky factorization " +
                                  "of the model discrepancy")
            tmp_dataspace_1 = cho_solve(LK, data.get_data())
        else:
            tmp_dataspace_1 = np.zeros(0)

        # interpolate to dataspace

        tmp_meshspace_1 = im.interp_data_to_mesh(tmp_dataspace_1)

        # solve forcing covariance and interpolate to dataspace

        tmp_meshspace_2 = _solve_forcing_covariance(G, A, tmp_meshspace_1)._scale(rho) + x.vector()

        tmp_dataspace_1 = im.interp_mesh_to_data(tmp_meshspace_2)

        if G.comm.rank == 0:
            try:
                L = cho_factor(Ks/rho**2 + Cu)
            except LinAlgError:
                raise LinAlgError("Error attempting to compute the Cholesky factorization " +
                                  "of the model discrepancy plus forcing covariance")
            tmp_dataspace_2 = cho_solve(L, tmp_dataspace_1)
        else:
            tmp_dataspace_2 = np.zeros(0)

        tmp_meshspace_1 = im.interp_data_to_mesh(tmp_dataspace_2)

        tmp_meshspace_1 = _solve_forcing_covariance(G, A, tmp_meshspace_1)

        x.assign((tmp_meshspace_2 - tmp_meshspace_1).function)

    # deallocate interpolation matrix

    im.destroy()

def solve_posterior_covariance(A, b, G, data, params, ensemble_comm=COMM_SELF):
    """
    solve for conditioned fem plus covariance in the data space

    returns solution as numpy arrays on the root process (rank 0)

    note that unlike the meshspace solver, this uses a return value rather than a
    Firedrake/PETSc style interface to create the solution. I was unable to get this
    to work by modifying the arrays in the function. This has the benefit of not requiring
    the user to pre-set the array sizes (the arrays are different sizes on the processes,
    as the solution is collected at the root of both the spatial comm and the ensemble comm)
    """

    if not isinstance(A, Matrix):
       raise TypeError("A must be a firedrake matrix")
    if not isinstance(b, (Function, Vector)):
        raise TypeError("b must be a firedrake function or vector")
    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a forcing covariance")
    if not isinstance(data, ObsData):
        raise TypeError("data must be an ObsData type")
    if not isinstance(ensemble_comm, type(COMM_WORLD)):
        raise TypeError("ensemble_comm must be an MPI communicator created with a firedrake Ensemble")

    params = np.array(params, dtype=np.float64)
    assert params.shape == (3,), "bad shape for model discrepancy parameters"
    rho = np.exp(params[0])

    # get prior solution

    muy, Cuy = solve_prior_covariance(A, b, G, data, ensemble_comm)

    if ensemble_comm.rank == 0 and G.comm.rank == 0:
        try:
            LK = cho_factor(data.calc_K_plus_sigma(params[1:]))
            Kinv = cho_solve(LK, np.eye(data.get_n_obs()))
            LC = cho_factor(Cuy)
            Cinv = cho_solve(LC, np.eye(data.get_n_obs()))
            L = cho_factor(Cinv + rho**2*Kinv)
            Cuy = cho_solve(L, np.eye(data.get_n_obs()))
        except LinAlgError:
            raise LinAlgError("Cholesky factorization of one of the covariance matrices failed")

        # get posterior mean

        muy = cho_solve(L, rho**2*cho_solve(LK, data.get_data()/rho) + cho_solve(LC, muy))

    return muy, Cuy

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

    if not isinstance(A, Matrix):
       raise TypeError("A must be a firedrake matrix")
    if not isinstance(b, (Function, Vector)):
        raise TypeError("b must be a firedrake function or vector")
    if not isinstance(G, ForcingCovariance):
        raise TypeError("G must be a forcing covariance")
    if not isinstance(data, ObsData):
        raise TypeError("data must be an ObsData type")
    if not isinstance(ensemble_comm, type(COMM_WORLD)):
        raise TypeError("ensemble_comm must be an MPI communicator created with a firedrake Ensemble")

    # create interpolation matrix

    im = InterpolationMatrix(G.function_space, data.get_coords())

    # form interpolated prior covariance across all ensemble processes

    Cu = im.interp_covariance_to_data(G, A, ensemble_comm)

    # solve base FEM (prior mean) and interpolate to data space on root

    x = Function(G.function_space)

    if ensemble_comm.rank == 0:
        solve(A, x, b)
        mu = im.interp_mesh_to_data(x.vector())
    else:
        mu = np.zeros(0)

    im.destroy()

    return mu, Cu