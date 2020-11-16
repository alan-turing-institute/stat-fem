"""
Solving module just provides functional wrappers to the LinearSolver class for one-off computations.
For repeated use, please use the LinearSolver class
"""

from firedrake import COMM_SELF
from .LinearSolver import LinearSolver

def solve_posterior(A, x, b, G, data, params, ensemble_comm=COMM_SELF, scale_mean=False, **kwargs):
    """
    Solve for the FEM posterior conditioned on the data

    Standalone function to solve for the posterior FEM solution on the mesh.
    Solution is stored in the provided Firedrake function ``x`` to be
    consistent with the Firedrake solving interface. This is just a
    wrapper to the ``solve_posterior`` method of the stat-fem
    ``LinearSolver`` class.

    Note that the solution is only stored in the root of the ensemble
    communicator if the forcing covariance solves are parallelized.
    The Firedrake function on other processes will not be modified.

    In addition to the parameters specified here, additional keyword
    arguments to control the PETSc solver options can be passed here
    and will be used when creating  the ``LinearSolver`` class.
    Please see the documentation for the ``LinearSolver`` class for
    more details.

    :param A: Assembled Firedrake Matrix for the FEM
    :type A: Firedrake Matrix
    :param x: Firedrake Function to hold the solution. Will be modified
              in place.
    :type x: Firedrake Function
    :param b: Assembled RHS vector, must be a Firedrake Vector
    :type b: Firedrake Vector
    :param G: Forcing covariance matrix
    :type G: ForcingCovariance
    :param params: Model discrepancy parameters to use in computing the
                   solution. Must by a 1D numpy array of length 3. Note
                   that these parameters are specified on a logarithmic
                   scale to enforce positivity.
    :type params: ndarray
    :param ensemble_comm: MPI communicator to use for parallelizing the
                          covariance solves. Optional, default is to
                          do the solves serially.
    :type ensemble_comm: MPI Communicator
    :param scale_mean: If true, scale the output by the model discrepancy
                       scaling factor. Optional, default is ``False``
    :type scale_mean: bool
    :param **kwargs: Additional keyword arguments for the PETSc solver.
    :returns: None
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm, **kwargs)
    ls.set_params(params)
    ls.solve_posterior(x, scale_mean)

def solve_posterior_covariance(A, b, G, data, params, ensemble_comm=COMM_SELF, scale_mean=False, **kwargs):
    """
    Solve for the FEM posterior conditioned on the data and the covariance

    Standalone function to solve for the posterior FEM solution
    interpolated to the sensor locations and the posterior covariance.
    Solution is returned as a pair of numpy arrays holding the solution
    in a 1D array and the covariance matrix in a 2D array. If the solves
    are parallelized, the non-root process arrays will contain empty arrays.
    This is just a wrapper to the ``solve_posterior_covariance`` method of
    the stat-fem ``LinearSolver`` class.

    In addition to the parameters specified here, additional keyword
    arguments to control the PETSc solver options can be passed here
    and will be used when creating  the ``LinearSolver`` class.
    Please see the documentation for the ``LinearSolver`` class for
    more details.

    :param A: Assembled Firedrake Matrix for the FEM
    :type A: Firedrake Matrix
    :param b: Assembled RHS vector, must be a Firedrake Vector
    :type b: Firedrake Vector
    :param G: Forcing covariance matrix
    :type G: ForcingCovariance
    :param data: ObsData object holding sensor data
    :type data: ObsData
    :param params: Model discrepancy parameters to use in computing the
                   solution. Must by a 1D numpy array of length 3. Note
                   that these parameters are specified on a logarithmic
                   scale to enforce positivity.
    :type params: ndarray
    :param ensemble_comm: MPI communicator to use for parallelizing the
                          covariance solves. Optional, default is to
                          do the solves serially.
    :type ensemble_comm: MPI Communicator
    :param scale_mean: If true, scale the output by the model discrepancy
                       scaling factor. Optional, default is ``False``
    :type scale_mean: bool
    :param **kwargs: Additional keyword arguments for the PETSc solver.
    :returns: Mean and covariance matrix of the posterior as numpy arrays.
              Mean is a 1D array and covariance is a 2D array.
    :rtype: tuple of 2 ndarrays
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm, **kwargs)
    ls.set_params(params)
    return ls.solve_posterior_covariance(scale_mean)

def solve_prior(A, b, G, data, ensemble_comm=COMM_SELF, **kwargs):
    """
    Solve for the FEM prior and covariance

    Standalone function to solve for the prior FEM solution
    interpolated to the sensor locations and the covariance.
    Solution is returned as a pair of numpy arrays holding the solution
    in a 1D array and the covariance matrix in a 2D array. If the solves
    are parallelized, the non-root process arrays will contain empty arrays.
    This is just a wrapper to the ``solve_prior`` method of
    the stat-fem ``LinearSolver`` class.

    Note that while the prior solution does not use the data, an
    ``ObsData`` object is still required to determine the sensor
    positions to which the solution should be interpolated. Any
    data and uncertainty information in the data object is ignored.

    In addition to the parameters specified here, additional keyword
    arguments to control the PETSc solver options can be passed here
    and will be used when creating  the ``LinearSolver`` class.
    Please see the documentation for the ``LinearSolver`` class for
    more details.

    :param A: Assembled Firedrake Matrix for the FEM
    :type A: Firedrake Matrix
    :param b: Assembled RHS vector, must be a Firedrake Vector
    :type b: Firedrake Vector
    :param G: Forcing covariance matrix
    :type G: ForcingCovariance
    :param data: ObsData object holding sensor data locations
    :type data: ObsData
    :param ensemble_comm: MPI communicator to use for parallelizing the
                          covariance solves. Optional, default is to
                          do the solves serially.
    :type ensemble_comm: MPI Communicator
    :param **kwargs: Additional keyword arguments for the PETSc solver.
    :returns: Mean and covariance matrix of the prior as numpy arrays.
              Mean is a 1D array and covariance is a 2D array.
    :rtype: tuple of 2 ndarrays
    """

    return LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm, **kwargs).solve_prior()

def solve_prior_generating(A, b, G, data, params, ensemble_comm=COMM_SELF, **kwargs):
    """
    Solve for the FEM prior of the generating process and covariance

    Standalone function to solve for the prior generating process
    of the FEM solution interpolated to the sensor locations and the
    covariance. Solution is returned as a pair of numpy arrays holding
    the solution in a 1D array and the covariance matrix in a 2D array.
    If the solves are parallelized, the non-root process arrays will
    contain empty arrays. This is just a wrapper to the
    ``solve_prior_generating`` method of the stat-fem ``LinearSolver``
    class.

    Note that while the prior solution does not use the data, an
    ``ObsData`` object is still required to determine the sensor
    positions to which the solution should be interpolated. Any
    data and uncertainty information in the data object is ignored.

    In addition to the parameters specified here, additional keyword
    arguments to control the PETSc solver options can be passed here
    and will be used when creating  the ``LinearSolver`` class.
    Please see the documentation for the ``LinearSolver`` class for
    more details.

    :param A: Assembled Firedrake Matrix for the FEM
    :type A: Firedrake Matrix
    :param b: Assembled RHS vector, must be a Firedrake Vector
    :type b: Firedrake Vector
    :param G: Forcing covariance matrix
    :type G: ForcingCovariance
    :param data: ObsData object holding sensor data locations
    :type data: ObsData
    :param params: Model discrepancy parameters to use in computing the
                   solution. Must by a 1D numpy array of length 3. Note
                   that these parameters are specified on a logarithmic
                   scale to enforce positivity.
    :type params: ndarray
    :param ensemble_comm: MPI communicator to use for parallelizing the
                          covariance solves. Optional, default is to
                          do the solves serially.
    :type ensemble_comm: MPI Communicator
    :param **kwargs: Additional keyword arguments for the PETSc solver.
    :returns: Mean and covariance matrix of the prior of the generating
              process as numpy arrays. Mean is a 1D array and covariance
              is a 2D array.
    :rtype: tuple of 2 ndarrays
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm, **kwargs)
    ls.set_params(params)
    return ls.solve_prior_generating()

def solve_posterior_generating(A, b, G, data, params, ensemble_comm=COMM_SELF, **kwargs):
    """
    Solve for the FEM posterior of the generating process and covariance

    Standalone function to solve for the posterior generating process
    of the FEM solution interpolated to the sensor locations and the
    covariance. Solution is returned as a pair of numpy arrays holding
    the solution in a 1D array and the covariance matrix in a 2D array.
    If the solves are parallelized, the non-root process arrays will
    contain empty arrays. This is just a wrapper to the
    ``solve_posterior_generating`` method of the stat-fem ``LinearSolver``
    class.

    In addition to the parameters specified here, additional keyword
    arguments to control the PETSc solver options can be passed here
    and will be used when creating  the ``LinearSolver`` class.
    Please see the documentation for the ``LinearSolver`` class for
    more details.

    :param A: Assembled Firedrake Matrix for the FEM
    :type A: Firedrake Matrix
    :param b: Assembled RHS vector, must be a Firedrake Vector
    :type b: Firedrake Vector
    :param G: Forcing covariance matrix
    :type G: ForcingCovariance
    :param data: ObsData object holding sensor data locations
    :type data: ObsData
    :param params: Model discrepancy parameters to use in computing the
                   solution. Must by a 1D numpy array of length 3. Note
                   that these parameters are specified on a logarithmic
                   scale to enforce positivity.
    :type params: ndarray
    :param ensemble_comm: MPI communicator to use for parallelizing the
                          covariance solves. Optional, default is to
                          do the solves serially.
    :type ensemble_comm: MPI Communicator
    :param **kwargs: Additional keyword arguments for the PETSc solver.
    :returns: Mean and covariance matrix of the posterior of the generating
              process as numpy arrays. Mean is a 1D array and covariance
              is a 2D array.
    :rtype: tuple of 2 ndarrays
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm, **kwargs)
    ls.set_params(params)
    return ls.solve_posterior_generating()

def predict_mean(A, b, G, data, params, coords, ensemble_comm=COMM_SELF, scale_mean=True, **kwargs):
    """
    Solve for the predicted mean FEM posterior values at new locations

    Standalone function to solve for the predicted posterior mean 
    FEM solution at new sensor locations. Solution is returned as a 1D numpy
    array holding the predicted values. If the solves
    are parallelized, the non-root process arrays will contain empty arrays.
    This is just a wrapper to the ``predict_mean`` method of
    the stat-fem ``LinearSolver`` class.

    Locations where predictions are desired are specified by ``coords``
    which is a 2D numpy array where the first index indicates the
    various prediction points and the second index indicates the spatial
    dimension.

    In addition to the parameters specified here, additional keyword
    arguments to control the PETSc solver options can be passed here
    and will be used when creating  the ``LinearSolver`` class.
    Please see the documentation for the ``LinearSolver`` class for
    more details.

    :param A: Assembled Firedrake Matrix for the FEM
    :type A: Firedrake Matrix
    :param b: Assembled RHS vector, must be a Firedrake Vector
    :type b: Firedrake Vector
    :param G: Forcing covariance matrix
    :type G: ForcingCovariance
    :param data: ObsData object holding sensor data
    :type data: ObsData
    :param params: Model discrepancy parameters to use in computing the
                   solution. Must by a 1D numpy array of length 3. Note
                   that these parameters are specified on a logarithmic
                   scale to enforce positivity.
    :type params: ndarray
    :param coords: Spatial locations where predictions will be made.
                   Must be a 2D numpy array where the first index
                   indicates the different points to be predicted and
                   the second index is the spatial coordinates.
    :type coords: ndarray
    :param ensemble_comm: MPI communicator to use for parallelizing the
                          covariance solves. Optional, default is to
                          do the solves serially.
    :type ensemble_comm: MPI Communicator
    :param scale_mean: If true, scale the output by the model discrepancy
                       scaling factor. Optional, default is ``True``
    :type scale_mean: bool
    :param **kwargs: Additional keyword arguments for the PETSc solver.
    :returns: Mean of the posterior prediction as a 1D numpy array.
    :rtype: ndarray
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm, **kwargs)
    ls.set_params(params)
    return ls.predict_mean(coords, scale_mean)


def predict_covariance(A, b, G, data, params, coords, unc, ensemble_comm=COMM_SELF, **kwargs):
    """
    Solve for the predicted covariance of the FEM posterior values at new locations

    Standalone function to solve for the predicted posterior covariance of the 
    FEM solution at new sensor locations. Solution is returned as a 2D numpy
    array holding the covariance matrix. If the solves
    are parallelized, the non-root process arrays will be an empty.
    This is just a wrapper to the ``predict_covariance`` method of
    the stat-fem ``LinearSolver`` class.

    Locations where predictions are desired are specified by ``coords``
    which is a 2D numpy array where the first index indicates the
    various prediction points and the second index indicates the spatial
    dimension. ``unc`` is the assumed uncertainty of hypothetical measurements
    taken at these locations and can either be scalar-like or array-like
    with the same length as ``coords``. All uncertainties must be
    non-negative.

    In addition to the parameters specified here, additional keyword
    arguments to control the PETSc solver options can be passed here
    and will be used when creating  the ``LinearSolver`` class.
    Please see the documentation for the ``LinearSolver`` class for
    more details.

    :param A: Assembled Firedrake Matrix for the FEM
    :type A: Firedrake Matrix
    :param b: Assembled RHS vector, must be a Firedrake Vector
    :type b: Firedrake Vector
    :param G: Forcing covariance matrix
    :type G: ForcingCovariance
    :param data: ObsData object holding sensor data
    :type data: ObsData
    :param params: Model discrepancy parameters to use in computing the
                   solution. Must by a 1D numpy array of length 3. Note
                   that these parameters are specified on a logarithmic
                   scale to enforce positivity.
    :type params: ndarray
    :param coords: Spatial locations where predictions will be made.
                   Must be a 2D numpy array where the first index
                   indicates the different points to be predicted and
                   the second index is the spatial coordinates.
    :type coords: ndarray
    :param unc: Uncertainty (as a standard deviation) of the hypothetical
                measurements, as a non-negative scalar or an array
                with the same length as ``coords`` of non-negative
                floats.
    :type unc: float or ndarray
    :param ensemble_comm: MPI communicator to use for parallelizing the
                          covariance solves. Optional, default is to
                          do the solves serially.
    :type ensemble_comm: MPI Communicator
    :param **kwargs: Additional keyword arguments for the PETSc solver.
    :returns: Covariance matrix of the posterior predictions as a 2D numpy
              array.
    :rtype: ndarray
    """

    ls = LinearSolver(A, b, G, data, ensemble_comm=ensemble_comm, **kwargs)
    ls.set_params(params)
    return ls.predict_covariance(coords, unc)
