import numpy as np
from firedrake import COMM_WORLD, COMM_SELF
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
    Solve for the FEM posterior conditioned on the data

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
    rho = params[0]
    assert rho > 0., "model/data scaling factor must be positive"

    # create interpolation matrix

    im = InterpolationMatrix(G.function_space, data.get_coords())

    # create numpy arrays for Ks matrix with appropriate sizes

    if ensemble_comm.rank == 0:
        Ks = data.calc_K_plus_sigma(params[1:])
    else:
        Ks = np.zeros((0,0))

    # first steps are done on root of ensemble only

    if ensemble_comm.rank == 0:

        # solve base FEM

        solve(A, x, b)

        # invert model discrepancy and interpolate into mesh space

        tmp_dataspace_1 = np.linalg.solve(Ks, data.get_data())
        tmp_meshspace_1 = im.interp_data_to_mesh(tmp_dataspace_1)

        # solve forcing covariance and interpolate to dataspace

        tmp_meshspace_2 = _solve_forcing_covariance(G, A, tmp_meshspace_1)+x.vector()
        tmp_dataspace_1 = im.interp_mesh_to_data(tmp_meshspace_2)

    else:

        # create dummy array for other ensemble members
        tmp_dataspace_1 = np.zeros(0)

    # solve model discrepancy plus forcing covariance system and interpolate into meshspace
    # (done on all ensemble processes)

    tmp_dataspace_2 = np.linalg.solve(Ks + im.interp_covariance_to_data(G, A, ensemble_comm),
                                      tmp_dataspace_1)

    # interpolate back onto FEM mesh

    if ensemble_comm.rank == 0:
        tmp_meshspace_1 = im.interp_data_to_mesh(tmp_dataspace_2)

    # deallocate interpolation matrix
    im.destroy()

    # solve final covariance system and place result in x
    # non-root ensemble processes have a dummy array

    if ensemble_comm.rank == 0:
        with x.dat.vec as solution:
            solution = (tmp_meshspace_2 - _solve_forcing_covariance(G, A, tmp_meshspace_1))

def solve_posterior_covariance(A, b, G, data, params, ensemble_comm=COMM_SELF):
    """
    solve for conditioned fem plus covariance in the data space

    note that unlike the meshspace solver, this uses a return value rather than a
    Firedrake/PETSc style interface to create the solution. I was unable to get this
    to work by modifying the arrays in the function. This has the benefit of not requiring
    the user to pre-set the array sizes (the arrays are different sizes on the processes,
    as the solution is collected at the root of both the spatial comm and the ensemble comm)
    """

    if not isinstance(A, Matrix):
       raise TypeError("A must be a firedrake matrix")
    if not isinstance(muy, np.ndarray):
        raise TypeError("muy must be a numpy array")
    if not isinstance(Cuy, np.ndarray):
        raise TypeError("Cuy must be a numpy array")
    assert muy.shape == (data.get_n_obs(),), "Bad shape for muy"
    assert Cuy.shape == (data.get_n_obs(), data.get_n_obs()), "Bad shape for Cuy"
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
    rho = params[0]
    assert rho > 0., "model/data scaling factor must be positive"

    # get prior solution

    muy, Cuy = solve_prior_covariance(A, b, G, data, params, ensemble_comm)

    if ensemble_comm.rank == 0:
        Kinv = np.linalg.inv(M.get_K_plus_sigma(params[1:]))
        Cinv = np.linalg.inv(Cuy)
        Cuy = np.linalg.inv(Cinv + Kinv)

        # get posterior mean

        muy = np.dot(Cuy, np.dot(Kinv, data.get_data()) + np.dot(Cinv, muy))

    return muy, Cuy

def solve_prior_covariance(A, b, G, data, params, ensemble_comm=COMM_SELF):
    """
    solve base (prior) fem plus covariance in the data space

    note that unlike the meshspace solver, this uses a return value rather than a
    Firedrake/PETSc style interface to create the solution. I was unable to get this
    to work by modifying the arrays in the function. This has the benefit of not requiring
    the user to pre-set the array sizes (the arrays are different sizes on the processes,
    as the solution is collected at the root of both the spatial comm and the ensemble comm)

    Note that since the data locations are needed to form the interpolation matrix and the
    scaling factor rho is needed to scale the model to observations, the observations and
    parameters are still required nonetheless.
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
    rho = params[0]
    assert rho > 0., "model/data scaling factor must be positive"

    # create interpolation matrix

    im = InterpolationMatrix(G.function_space, data.get_coords())

    # solve base FEM (prior mean) and interpolate to data space

    x = Function(G.function_space)

    if ensemble_comm.rank == 0:
        solve(A, x, b)
        mu = rho*im.interp_mesh_to_data(x.vector())
    else:
        mu = np.zeros(0)

    # form interpolated prior covariance and solve for posterior covariance

    Cu = rho**2*im.interp_covariance_to_data(G, A, ensemble_comm)

    im.destroy()

    return mu, Cu