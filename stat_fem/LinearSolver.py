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

class LinearSolver(object):
    "Class encapsulating all solves on the same set of model ingredients as methods"
    def __init__(self, A, b, G, data, priors=[None, None, None], ensemble_comm=COMM_SELF):
        "Create a new linear solver for statistical FEM problems"

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

    def __del__(self):
        "deallocates interpolation matrix"

        self.im.destroy()

    def set_params(self, params):
        "sets parameter values"

        params = np.array(params, dtype=np.float64)
        assert params.shape == (3,), "bad shape for model discrepancy parameters"

        self.params = params

    def solve_prior(self):
        """
        solve base (prior) fem plus covariance interpolated to the data locations

        returns solution as numpy arrays on the root process (rank 0) and caches the values
        for use in further computations (again on the root process)

        note that unlike the meshspace solver, this uses a return value rather than a
        Firedrake/PETSc style interface to create the solution. I was unable to get this
        to work by modifying the arrays in the function. This has the benefit of not requiring
        the user to pre-set the array sizes (the arrays are different sizes on the processes,
        as the solution is collected at the root of both the spatial comm and the ensemble comm)

        Note that since the data locations are needed, this still requires an ObsData object.
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

    def solve_posterior(self, x):
        """
        Solve for the FEM posterior conditioned on the data. solution is stored in the
        provided firedrake function x

        Note that the solution is only stored in the root of the ensemble comm if the
        forcing covariance solves are parallelized. The Firedrake function on other
        processes will not be modified.
        """

        # create interpolation matrix if not cached

        if self.Cu is None:
            self.solve_prior()

        if self.params is None:
            raise ValueError("must set parameter values to solve posterior")

        rho = np.exp(self.params[0])

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

            x.assign((tmp_meshspace_2 - tmp_meshspace_1).function)


    def solve_posterior_covariance(self):
        """
        solve for conditioned fem plus covariance in the data space

        returns solution as numpy arrays on the root process (rank 0)

        note that unlike the meshspace solver, this uses a return value rather than a
        Firedrake/PETSc style interface to create the solution. I was unable to get this
        to work by modifying the arrays in the function. This has the benefit of not requiring
        the user to pre-set the array sizes (the arrays are different sizes on the processes,
        as the solution is collected at the root of both the spatial comm and the ensemble comm)
        """

        # create interpolation matrix if not cached

        if self.mu is None or self.Cu is None:
            self.solve_prior()

        if self.params is None:
            raise ValueError("must set parameter values to solve posterior")

        rho = np.exp(self.params[0])

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

        return muy, Cuy

    def solve_prior_generating(self):
        "solve for the prior of the generating process"

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
        "solve for the posterior of the generating process"

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

    def predict_mean(self, coords):
        """
        predict mean data values at unmeasured locations

        returns vector of predicted sensor values on root process as numpy array. requires
        only a small overhead above the computational work of finding the posterior mean
        (i.e. get mean value at new sensor locations for "free" once you have solved the
        posterior)
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

        x = Function(self.G.function_space)

        self.solve_posterior(x)

        im = InterpolationMatrix(self.G.function_space, coords)

        mu = rho*im.interp_mesh_to_data(x.vector())

        im.destroy()

        return mu

    def predict_covariance(self, coords, unc):
        """
        predict the mean and covariance of data values at unmeasured locations

        returns vector of predicted sensor values on root process as numpy array. requires
        doing an additional 2*n_pred FEM solves to get the full covariance at the new locations.
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


