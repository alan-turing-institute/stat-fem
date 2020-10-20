import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import LinAlgError
from firedrake import COMM_WORLD, COMM_SELF
from firedrake.function import Function
from firedrake.matrix import Matrix
from firedrake.vector import Vector
from firedrake.solving import solve
from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix
from .ObsData import ObsData
from .solving_utils import solve_forcing_covariance, interp_covariance_to_data

class LinearSolver(object):
    r"""
    Class encapsulating all solves on the same FEM model

    This class forms the base of all Stat FEM computations for a linear problem. It requires
    the base FEM problem, the forcing covariance (represented by a ``ForcingCovariance`` object),
    the sensor locations, data, and uncertanties (represented by a ``ObsData`` object),
    priors on the model discrepancy hyperparameters (optional), and an ensemble MPI communicator
    for parallelizing the covariance solves (optional).

    Once these are set, the prior solves can be done and cached, which is generally the most
    computationally expensive part of the modeling. The class also contains methods for
    performing parameter estimation (a ``logposterior`` to compute the negative log posterior
    or marginal likelihood if no priors are specified and its associated derivatives), and
    prediction of sensor values and uncertainties at unmeasured locations.

    :ivar A: the FEM stiffness matrix
    :type A: Firedrake Matrix
    :ivar b: the FEM RHS vector
    :type b: Firedrake Vector or Function
    :ivar G: Forcing Covariance sparse matrix
    :type G: ForcingCovariance
    :ivar data: Sensor locations, observed values, and uncertainties
    :type data: ObsData
    :ivar priors: list of prior distributions on hyperparameters or all ``None`` if uninformative
                  priors are assumed
    :type priors: list
    :ivar ensemble_comm: Firedrake Ensemble communicator for parallelizing covariance solves
    :type ensemble_comm: MPI Communicator
    :ivar params: Current set of parameters (a numpy array of length 3) representing the
                  data/model scaling factor :math:`{\rho}`, model discrepancy covariance,
                  and model discrepancy correlation length. All parameters are on a logarithmic
                  scale to enforce positivity.
    :type params: ndarray
    :ivar im: interpolation matrix used to interpolate between FEM mesh and sensor data
    :type im: InterpolationMatrix
    :ivar x: Prior FEM solution on distributed FEM mesh
    :type x: Firedrake Function
    :ivar mu: Prior FEM solution interpolated to sensor locations on root process (other processes
              have arrays of length 0)
    :type mu: ndarray
    :ivar Cu: Prior FEM covariance interpolated to sensor locations on root process (other processes
              have arrays of shape ``(0, 0)``
    :type Cu: ndarray
    :ivar current_logpost: Current value of the negative log-posterior (or log likelihood if prior
                           is uninformative)
    :type current_logpost: float
    """

    def __init__(self, A, b, G, data, priors=[None, None, None], ensemble_comm=COMM_SELF):
        r"""
        Create a new object encapsulating all solves on the same FEM model

        Initialize a new object for a given FEM problem to perform the Stat FEM solves.

        This class forms the base of all Stat FEM computations for a linear problem. It requires
        the base FEM problem, the forcing covariance (represented by a ``ForcingCovariance`` object),
        the sensor locations, data, and uncertanties (represented by a ``ObsData`` object),
        priors on the model discrepancy hyperparameters (optional), and an ensemble MPI communicator
        for parallelizing the covariance solves (optional).

        :param A: the FEM stiffness matrix
        :type A: Firedrake Matrix
        :param b: the FEM RHS vector
        :type b: Firedrake Vector or Function
        :param G: Forcing Covariance sparse matrix
        :type G: ForcingCovariance
        :param data: Sensor locations, observed values, and uncertainties
        :type data: ObsData
        :param priors: list of prior distributions on hyperparameters or all ``None`` if uninformative
                       priors are assumed (optional)
        :type priors: list
        :param ensemble_comm: Firedrake Ensemble communicator for parallelizing covariance solves (optional)
        :type ensemble_comm: MPI Communicator
        :returns: new ``LinearSolver`` instance
        :rtype: LinearSolver
        """

        if not isinstance(A, Matrix):
           raise TypeError("A must be a firedrake matrix")
        if not isinstance(b, (Function, Vector)):
            raise TypeError("b must be a firedrake function or vector")
        if not isinstance(G, ForcingCovariance):
            raise TypeError("G must be a forcing covariance")
        if not isinstance(data, ObsData):
            raise TypeError("data must be an ObsData type")
        if not isinstance(priors, list):
            raise TypeError("priors must be a list of prior objects or None")
        if not len(priors) == 3:
            raise ValueError("priors must be a list of prior objects or None of length 3")
        for p in priors:
            if not p is None:
                raise TypeError("priors must be a list of prior objects or None")
        if not isinstance(ensemble_comm, type(COMM_WORLD)):
            raise TypeError("ensemble_comm must be an MPI communicator created with a firedrake Ensemble")

        self.A = A
        self.b = b
        self.G = G
        self.data = data
        self.ensemble_comm = ensemble_comm
        self.priors = list(priors)
        self.params = None

        self.im = InterpolationMatrix(G.function_space, data.get_coords())

        self.x = None
        self.mu = None
        self.Cu = None
        self.current_logpost = None

    def __del__(self):
        r"""
        Delete the LinearSolver object

        When deleting a LinearSolver, one needs to deallocate the memory for the interpolation
        matrix. No inputs or return values.
        """

        self.im.destroy()

    def set_params(self, params):
        r"""
        Sets parameter values

        Checks and sets new values of the hyperparameters. New parameters must be a numpy
        array of length 3. First parameter is the data/model scaling factor :math:`{\rho}`,
        second parameter is the model discrepancy covariance, and the third parameter is
        the model discrepancy correlation length. All parameters are assumed to be on a
        logarithmic scale to enforce positivity.

        :param params: New set of parameters (must be a numpy array of length 3)
        :type params: ndarray
        :returns: None
        """

        params = np.array(params, dtype=np.float64)
        assert params.shape == (3,), "bad shape for model discrepancy parameters"

        self.params = params

    def solve_prior(self):
        r"""
        Solve base (prior) FEM plus covariance interpolated to the data locations

        This method solves the prior FEM and covariance interpolated to the sensor locations.
        It does not require setting parameter values, as the model discrepancy does not
        influence these results. The covariance is cached as it is expensive to compute
        and is re-used in all other solves.

        In addition to caching the results, the method returns solution as numpy arrays
        on the root process (rank 0).

        Note that unlike the solve done in the meshspace, this uses a return value rather than a
        Firedrake/PETSc style interface to place the solution in a pre-allocated ``Function``.
        This is because each process has a different array size, so would require correctly
        pre-allocating arrays of different lengths on each process.

        :returns: FEM prior mean and covariance (as a tuple of numpy arrays) on the root process.
                  Non-root processes return numpy arrays of shape ``(0,)`` (mean) and ``(0, 0)``
                  (covariance).
        :rtype: tuple of ndarrays
        """

        # form interpolated prior covariance across all ensemble processes

        self.Cu = interp_covariance_to_data(self.im, self.G, self.A, self.im, self.ensemble_comm)

        # solve base FEM (prior mean) and interpolate to data space on root

        self.x = Function(self.G.function_space)

        if self.ensemble_comm.rank == 0:
            solve(self.A, self.x, self.b)
            self.mu = self.im.interp_mesh_to_data(self.x.vector())
        else:
            self.mu = np.zeros(0)

        return self.mu, self.Cu

    def solve_posterior(self, x, scale_mean=False):
        r"""
        Solve FEM posterior in mesh space

        Solve for the FEM posterior conditioned on the data on the FEM mesh. The solution
        is stored in the preallocated Firedrake ``Function``.

        Note that if an ensemble communicator was used to parallelize the covariance solves,
        the solution is only stored in the root of the ensemble communicator. The Firedrake
        ``Function`` on the other processes will not be modified.

        The optional ``scale_mean`` argument determines if the solution is to be re-scaled
        by the model discrepancy scaling factor. This value is by default ``False``.
        To re-scale to match the data, pass ``scale_mean=True``.

        :param x: Firedrake ``Function`` for holding the solution. This is modified in place
                  by the method.
        :type x: Firedrake Function
        :param scale_mean: Boolean indicating if the mean should be scaled by the model
                           discrepancy scaling factor. Optional, default is ``False``
        :type scale_mean: bool
        :returns: None
        """

        if not isinstance(bool(scale_mean), bool):
            raise TypeError("scale_mean argument must be boolean-like")
        
        # create interpolation matrix if not cached

        if self.Cu is None:
            self.solve_prior()

        if self.params is None:
            raise ValueError("must set parameter values to solve posterior")

        rho = np.exp(self.params[0])

        if scale_mean:
            scalefact = rho
        else:
            scalefact = 1.
        
        # remaining solves are just done on ensemble root

        if self.ensemble_comm.rank == 0:

            if self.G.comm.rank == 0:
                Ks = self.data.calc_K_plus_sigma(self.params[1:])
                try:
                    LK = cho_factor(Ks)
                except LinAlgError:
                    raise LinAlgError("Error attempting to compute the Cholesky factorization " +
                                      "of the model discrepancy")
                tmp_dataspace_1 = cho_solve(LK, self.data.get_data())
            else:
                tmp_dataspace_1 = np.zeros(0)

            # interpolate to dataspace

            tmp_meshspace_1 = self.im.interp_data_to_mesh(tmp_dataspace_1)

            # solve forcing covariance and interpolate to dataspace

            tmp_meshspace_2 = solve_forcing_covariance(self.G, self.A, tmp_meshspace_1)._scale(rho) + self.x.vector()

            tmp_dataspace_1 = self.im.interp_mesh_to_data(tmp_meshspace_2)

            if self.G.comm.rank == 0:
                try:
                    L = cho_factor(Ks + rho**2*self.Cu)
                except LinAlgError:
                    raise LinAlgError("Error attempting to compute the Cholesky factorization " +
                                      "of the model discrepancy plus forcing covariance")
                tmp_dataspace_2 = cho_solve(L, tmp_dataspace_1)
            else:
                tmp_dataspace_2 = np.zeros(0)

            tmp_meshspace_1 = self.im.interp_data_to_mesh(tmp_dataspace_2)

            tmp_meshspace_1 = solve_forcing_covariance(self.G, self.A, tmp_meshspace_1)._scale(rho**2)

            x.assign((tmp_meshspace_2 - tmp_meshspace_1)._scale(scalefact).function)


    def solve_posterior_covariance(self, scale_mean=False):
        r"""
        Solve posterior FEM and covariance interpolated to the data locations

        This method solves the posterior FEM and covariance interpolated to the sensor
        locations. The method returns solution as numpy arrays on the root process (rank 0).

        Note that unlike the solve done in the meshspace, this uses a return value rather than a
        Firedrake/PETSc style interface to place the solution in a pre-allocated ``Function``.
        This is because each process has a different array size, so would require correctly
        pre-allocating arrays of different lengths on each process.

        The optional ``scale_mean`` argument determines if the solution is to be re-scaled
        by the model discrepancy scaling factor. This value is by default ``False``.
        To re-scale to match the data, pass ``scale_mean=True``.

        :returns: FEM posterior mean and covariance (as a tuple of numpy arrays) on the root process.
                  Non-root processes return numpy arrays of shape ``(0,)`` (mean) and ``(0, 0)``
                  (covariance).
        :param scale_mean: Boolean indicating if the mean should be scaled by the model
                           discrepancy scaling factor. Optional, default is ``False``
        :type scale_mean: bool
        :rtype: tuple of ndarrays
        """

        if not isinstance(bool(scale_mean), bool):
            raise TypeError("scale_mean argument must be boolean-like")
        
        # create interpolation matrix if not cached

        if self.mu is None or self.Cu is None:
            self.solve_prior()

        if self.params is None:
            raise ValueError("must set parameter values to solve posterior")

        rho = np.exp(self.params[0])

        if scale_mean:
            scalefact = rho
        else:
            scalefact = 1.
        
        if self.ensemble_comm.rank == 0 and self.G.comm.rank == 0:
            try:
                Ks = self.data.calc_K_plus_sigma(self.params[1:])
                LK = cho_factor(Ks)
                LC = cho_factor(Ks + rho**2*self.Cu)
            except LinAlgError:
                raise LinAlgError("Cholesky factorization of one of the covariance matrices failed")

            # compute posterior mean

            muy = rho*np.dot(self.Cu, cho_solve(LK, self.data.get_data())) + self.mu
            muy_tmp = rho**2*np.dot(self.Cu, cho_solve(LC, muy))
            muy = muy - muy_tmp

            # compute posterior covariance

            Cuy = self.Cu - rho**2*np.dot(self.Cu, cho_solve(LC, self.Cu))

        else:
            muy = np.zeros(0)
            Cuy = np.zeros((0,0))

        return scalefact*muy, Cuy

    def solve_prior_generating(self):
        r"""
        Solve for the prior of the generating process

        This method solves for the prior of the generating process before looking at the data.
        The main computational cost is solving for the prior of the covariance, so if this is
        cached from a previous solve this is a simple calculation.

        :returns: FEM prior mean and covariance of the true generating process (as a tuple of
                  numpy arrays) on the root process. Non-root processes return numpy arrays of
                  shape ``(0,)`` (mean) and ``(0, 0)`` (covariance).
        :rtype: tuple of ndarrays
        """

        # create interpolation matrix if not cached

        if self.mu is None or self.Cu is None:
            self.solve_prior()

        if self.params is None:
            raise ValueError("must set parameter values to solve prior of generating process")

        rho = np.exp(self.params[0])

        if self.G.comm.rank == 0 and self.ensemble_comm.rank == 0:
            m_eta = rho*self.mu
            C_eta = rho**2*self.Cu + self.data.calc_K(self.params[1:])
        else:
            m_eta = np.zeros(0)
            C_eta = np.zeros((0,0))

        return m_eta, C_eta

    def solve_posterior_generating(self):
        r"""
        Solve for the posterior of the generating process

        This method solves for the posterior of the generating process before looking at the data.
        The main computational cost is solving for the prior of the covariance, so if this is
        cached from a previous solve this is a simple calculation.

        :returns: FEM posterior mean and covariance of the true generating process (as a tuple of
                  numpy arrays) on the root process. Non-root processes return numpy arrays of
                  shape ``(0,)`` (mean) and ``(0, 0)`` (covariance).
        :rtype: tuple of ndarrays
        """

        # create interpolation matrix if not cached

        m_eta, C_eta = self.solve_prior_generating()

        if self.ensemble_comm.rank == 0 and self.G.comm.rank == 0:
            try:
                L = cho_factor(self.data.get_unc()**2*np.eye(self.data.get_n_obs()) + C_eta)
            except LinAlgError:
                raise LinAlgError("Cholesky factorization of the covariance matrix failed")

            C_etay = cho_solve(L, self.data.get_unc()**2*C_eta)

            m_etay = cho_solve(L, np.dot(C_eta, self.data.get_data()) + self.data.get_unc()**2*m_eta)
        else:
            m_etay = np.zeros(0)
            C_etay = np.zeros((0,0))

        return m_etay, C_etay

    def predict_mean(self, coords, scale_mean=True):
        r"""
        Compute the predictive mean

        This method computes the predictive mean of data values at unmeasured locations. It returns
        the vector of predicted sensor values on the root process as numpy array. It requires only a
        small overhead above the computational work of finding the posterior mean (i.e. you get
        the mean value at new sensor locations for "free" once you have solved the posterior).

        The optional ``scale_mean`` argument determines if the solution is to be re-scaled
        by the model discrepancy scaling factor. This value is by default ``True``.
        To re-scale to match the FEM solution, pass ``scale_mean=False``.

        :param coords: Spatial coordinates at which the mean will be predicted. Must be a
                       2D Numpy array (or a 1D array, which will assume the second axis has length
                       1)
        :type coords: ndarray
        :param scale_mean: Boolean indicating if the mean should be scaled by the model
                           discrepancy scaling factor. Optional, default is ``True``
        :type scale_mean: bool
        :returns: FEM prediction at specified sensor locations as a numpy array on the root process.
                  All other processes will have a numpy array of length 0.
        :rtype: ndarray
        """
        
        if not isinstance(bool(scale_mean), bool):
            raise TypeError("scale_mean argument must be boolean-like")

        coords = np.array(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = np.reshape(coords, (-1, 1))
        assert coords.ndim == 2, "coords must be a 1d or 2d array"
        assert coords.shape[1] == self.data.get_n_dim(), "axis 1 of coords must be the same length as the FEM dimension"

        if self.Cu is None:
            self.solve_prior()

        if self.params is None:
            raise ValueError("must set parameter values to make predictions")

        rho = np.exp(self.params[0])

        if scale_mean:
            scalefact = rho
        else:
            scalefact = 1.
        
        x = Function(self.G.function_space)

        self.solve_posterior(x)

        im = InterpolationMatrix(self.G.function_space, coords)

        mu = scalefact*im.interp_mesh_to_data(x.vector())

        im.destroy()

        return mu

    def predict_covariance(self, coords, unc):
        r"""
        Compute the predictive covariance

        This method computes the predictive covariance of data values at unmeasured locations.
        It returns the array of predicted sensor value covariances on the root process as numpy
        array. Unlike the mean, the predictive covariance requires doing two additional sets of
        covariance solves: one on the new sensor locations (to get the baseline covariance),
        and one set of solves that interpolates between the predictive points and the original
        sensor locations. This can be thought of as doing the covariance solves at the new points
        to get a baseline uncertainty, and then the cross-solves determine if any of the sensor
        data is close enough to the predictive points to reduce this uncertainty.

        :param coords: Spatial coordinates at which the mean will be predicted. Must be a
                       2D Numpy array (or a 1D array, which will assume the second axis has length
                       1)
        :type coords: ndarray
        :param unc: Uncertainty for unmeasured sensor locations (i.e. the statistical error one would
                    expect if these measurements were made). Can be a single non-negative float,
                    or an array of non-negative floats with the same length as the first axis of
                    ``coords``.
        :type unc: float or ndarray
        :returns: FEM predictive covariance at specified sensor locations as a numpy array on the
                  root process. All other processes will have a numpy array of shape ``(0, 0)``.
        :rtype: ndarray
        """

        coords = np.array(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = np.reshape(coords, (-1, 1))
        assert coords.ndim == 2, "coords must be a 1d or 2d array"
        assert coords.shape[1] == self.data.get_n_dim(), "axis 1 of coords must be the same length as the FEM dimension"

        if self.Cu is None:
            self.solve_prior()

        if self.params is None:
            raise ValueError("must set parameter values to make predictions")

        rho = np.exp(self.params[0])

        im_coords = InterpolationMatrix(self.G.function_space, coords)

        if coords.shape[0] > self.data.get_n_obs():
            Cucd = interp_covariance_to_data(im_coords, self.G, self.A, self.im, self.ensemble_comm)
        else:
            Cucd = interp_covariance_to_data(self.im, self.G, self.A, im_coords, self.ensemble_comm).T
        Cucc = interp_covariance_to_data(im_coords, self.G, self.A, im_coords, self.ensemble_comm)

        if self.ensemble_comm.rank == 0 and self.G.comm.rank == 0:
            try:
                Ks = self.data.calc_K_plus_sigma(self.params[1:])
                LC = cho_factor(Ks + rho**2*self.Cu)
            except LinAlgError:
                raise LinAlgError("Cholesky factorization of one of the covariance matrices failed")

            # compute predictive covariance

            Cuy = Cucc - rho**2*np.dot(Cucd, cho_solve(LC, Cucd.T))

            Cuy = ObsData(coords, np.zeros(coords.shape[0]), unc).calc_K_plus_sigma(self.params[1:]) + rho**2*Cuy

        else:
            Cuy = np.zeros((0,0))

        im_coords.destroy()

        return Cuy

    def logposterior(self, params):
        r"""
        Compute the negative log posterior for a particular set of parameters

        Computes the negative log posterior (negative marginal log-likelihood minus any
        prior log-probabilities). This is computed on the root process and then broadcast
        to all processes.

        The main computational expense is computing the prior mean and covariance, which only
        needs to be done once and can be cached. This also requires computing the Cholesky
        decomposition of the covariance plus model discrepancy.

        New parameters must be a numpy array of length 3. First parameter is the data/model
        scaling factor :math:`{\rho}`, second parameter is the model discrepancy covariance,
        and the third parameter is the model discrepancy correlation length. All parameters
        are assumed to be on a logarithmic scale to enforce positivity.

        :param params: New set of parameters (must be a numpy array of length 3)
        :type params: ndarray
        :returns: negative log posterior
        :rtype: float
        """

        self.set_params(params)
        rho = np.exp(self.params[0])

        if self.Cu is None or self.mu is None:
            self.solve_prior()

        # compute log-likelihood on root process and broadcast

        if COMM_WORLD.rank == 0:
            KCu = rho**2*self.Cu + self.data.calc_K_plus_sigma(self.params[1:])
            try:
                L = cho_factor(KCu)
            except LinAlgError:
                raise LinAlgError("Error attempting to factorize the covariance matrix " +
                                  "in model_loglikelihood")
            invKCudata = cho_solve(L, self.data.get_data() - rho*self.mu)
            log_posterior = 0.5*(self.data.get_n_obs()*np.log(2.*np.pi) +
                                 2.*np.sum(np.log(np.diag(L[0]))) +
                                 np.dot(self.data.get_data() - rho*self.mu, invKCudata))
            for i in range(3):
                if not self.priors[i] is None:
                    log_posterior -= self.priors[i].logp(self.params[i])
        else:
            log_posterior = None

        log_posterior = COMM_WORLD.bcast(log_posterior, root=0)

        assert not log_posterior is None, "error in broadcasting the log likelihood"

        COMM_WORLD.barrier()

        return log_posterior

    def logpost_deriv(self, params):
        r"""
        Compute the gradient of the negative log posterior for a particular set of parameters

        Computes the gradient of the negative log posterior (negative marginal log-likelihood
        minus any prior log-probabilities). This is computed on the root process and then broadcast
        to all processes.

        The main computational expense is computing the prior mean and covariance, which only
        needs to be done once and can be cached. This also requires computing the Cholesky
        decomposition of the covariance plus model discrepancy.

        New parameters must be a numpy array of length 3. First parameter is the data/model
        scaling factor :math:`{\rho}`, second parameter is the model discrepancy covariance,
        and the third parameter is the model discrepancy correlation length. All parameters
        are assumed to be on a logarithmic scale to enforce positivity.

        The returned log posterior gradient is a numpy array of length 3, with each component
        corresponding to the derivative of each of the input parameters.

        :param params: New set of parameters (must be a numpy array of length 3)
        :type params: ndarray
        :returns: gradient of the negative log posterior
        :rtype: ndarray
        """

        self.set_params(params)
        rho = np.exp(self.params[0])

        if self.Cu is None or self.mu is None:
            self.solve_prior()

        # compute log-likelihood on root process

        if COMM_WORLD.rank == 0:
            KCu = rho**2*self.Cu + self.data.calc_K_plus_sigma(params[1:])
            try:
                L = cho_factor(KCu)
            except LinAlgError:
                raise LinAlgError("Error attempting to factorize the covariance matrix " +
                                  "in model_loglikelihood")
            invKCudata = cho_solve(L, self.data.get_data() - rho*self.mu)

            K_deriv = self.data.calc_K_deriv(self.params[1:])

            deriv = np.zeros(3)

            deriv[0] = (-rho*np.dot(self.mu, invKCudata) -
                        rho**2*np.linalg.multi_dot([invKCudata, self.Cu, invKCudata]) +
                        rho**2*np.trace(cho_solve(L, self.Cu)))
            for i in range(0, 2):
                deriv[i + 1] = -0.5*(np.linalg.multi_dot([invKCudata, K_deriv[i], invKCudata]) -
                                    np.trace(cho_solve(L, K_deriv[i])))

            for i in range(3):
                if not self.priors[i] is None:
                    deriv[i] -= self.priors[i].dlogpdtheta(self.params[i])
        else:
            deriv = None

        deriv = COMM_WORLD.bcast(deriv, root=0)

        assert not deriv is None, "error in broadcasting the log likelihood derivative"

        COMM_WORLD.barrier()

        return deriv
