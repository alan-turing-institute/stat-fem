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
from .solving_utils import solve_forcing_covariance, interp_covariance_to_data

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

    Cu = interp_covariance_to_data(im, G, A, im, ensemble_comm)

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

        tmp_meshspace_2 = solve_forcing_covariance(G, A, tmp_meshspace_1)._scale(rho) + x.vector()

        tmp_dataspace_1 = im.interp_mesh_to_data(tmp_meshspace_2)

        if G.comm.rank == 0:
            try:
                L = cho_factor(Ks + rho**2*Cu)
            except LinAlgError:
                raise LinAlgError("Error attempting to compute the Cholesky factorization " +
                                  "of the model discrepancy plus forcing covariance")
            tmp_dataspace_2 = cho_solve(L, tmp_dataspace_1)
        else:
            tmp_dataspace_2 = np.zeros(0)

        tmp_meshspace_1 = im.interp_data_to_mesh(tmp_dataspace_2)

        tmp_meshspace_1 = solve_forcing_covariance(G, A, tmp_meshspace_1)._scale(rho**2)

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
            Ks = data.calc_K_plus_sigma(params[1:])
            LK = cho_factor(Ks)
            LC = cho_factor(Ks + rho**2*Cuy)
        except LinAlgError:
            raise LinAlgError("Cholesky factorization of one of the covariance matrices failed")

        # compute posterior mean

        muy = rho*np.dot(Cuy, cho_solve(LK, data.get_data())) + muy
        muy_tmp = rho**2*np.dot(Cuy, cho_solve(LC, muy))
        muy = muy - muy_tmp

        # compute posterior covariance

        Cuy = Cuy - rho**2*np.dot(Cuy, cho_solve(LC, Cuy))

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

    Cu = interp_covariance_to_data(im, G, A, im, ensemble_comm)

    # solve base FEM (prior mean) and interpolate to data space on root

    x = Function(G.function_space)

    if ensemble_comm.rank == 0:
        solve(A, x, b)
        mu = im.interp_mesh_to_data(x.vector())
    else:
        mu = np.zeros(0)

    im.destroy()

    return mu, Cu

def solve_prior_generating(A, b, G, data, params, ensemble_comm=COMM_SELF):
    "solve for the prior of the generating process"

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

    mu, Cu  = solve_prior_covariance(A, b, G, data, ensemble_comm)

    if G.comm.rank == 0 and ensemble_comm.rank == 0:
        m_eta = rho*mu
        C_eta = rho**2*Cu + data.calc_K(params[1:])
    else:
        m_eta = np.zeros(0)
        C_eta = np.zeros((0,0))

    return m_eta, C_eta

def solve_posterior_generating(A, b, G, data, params, ensemble_comm=COMM_SELF):
    "solve for the posterior of the generating process"

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

    m_eta, C_eta = solve_prior_generating(A, b, G, data, params, ensemble_comm)

    if ensemble_comm.rank == 0 and G.comm.rank == 0:
        try:
            L = cho_factor(data.get_unc()**2*np.eye(data.get_n_obs()) + C_eta)
        except LinAlgError:
            raise LinAlgError("Cholesky factorization of the covariance matrix failed")

        C_etay = cho_solve(L, data.get_unc()**2*C_eta)

        m_etay = cho_solve(L, np.dot(C_eta, data.get_data()) + data.get_unc()**2*m_eta)
    else:
        m_etay = np.zeros(0)
        C_etay = np.zeros((0,0))

    return m_etay, C_etay

def predict_mean(A, b, G, data, params, coords, ensemble_comm=COMM_SELF):
    """
    predict mean data values at unmeasured locations

    returns vector of predicted sensor values on root process as numpy array. requires
    only a small overhead above the computational work of finding the posterior mean
    (i.e. get mean value at new sensor locations for "free" once you have solved the
    posterior)
    """
    rho = np.exp(params[0])

    x = Function(G.function_space)

    solve_posterior(A, x, b, G, data, params, ensemble_comm)

    im = InterpolationMatrix(G.function_space, coords)

    return rho*im.interp_mesh_to_data(x.vector())

def predict_covariance(A, b, G, data, params, coords, unc, ensemble_comm=COMM_SELF):
    """
    predict the mean and covariance of data values at unmeasured locations

    returns vector of predicted sensor values on root process as numpy array. requires
    doing an additional 2*n_pred FEM solves to get the full covariance at the new locations.
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

    coords = np.array(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = np.reshape(coords, (-1, 1))
    assert coords.ndim == 2, "coords must be a 1d or 2d array"
    assert coords.shape[1] == data.get_n_dim(), "axis 1 of coords must be the same length as the FEM dimension"

    params = np.array(params, dtype=np.float64)
    assert params.shape == (3,), "bad shape for model discrepancy parameters"
    rho = np.exp(params[0])

    im_data = InterpolationMatrix(G.function_space, data.get_coords())

    im_coords = InterpolationMatrix(G.function_space, coords)

    Cu   = interp_covariance_to_data(im_data, G, A, im_data, ensemble_comm)
    if coords.shape[0] > data.get_n_obs():
        Cucd = interp_covariance_to_data(im_coords, G, A, im_data, ensemble_comm)
    else:
        Cucd = interp_covariance_to_data(im_data, G, A, im_coords, ensemble_comm).T
    Cucc = interp_covariance_to_data(im_coords, G, A, im_coords, ensemble_comm)

    if ensemble_comm.rank == 0 and G.comm.rank == 0:
        try:
            Ks = data.calc_K_plus_sigma(params[1:])
            LC = cho_factor(Ks + rho**2*Cu)
        except LinAlgError:
            raise LinAlgError("Cholesky factorization of one of the covariance matrices failed")

        # compute predictive covariance

        Cuy = Cucc - rho**2*np.dot(Cucd, cho_solve(LC, Cucd.T))

        Cuy = ObsData(coords, np.zeros(coords.shape[0]), unc).calc_K_plus_sigma(params[1:]) + rho**2*Cuy

    else:
        Cuy = np.zeros((0,0))

    return Cuy