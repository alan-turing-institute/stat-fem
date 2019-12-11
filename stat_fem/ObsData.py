import numpy as np
from firedrake import COMM_WORLD, COMM_SELF
from .CovarianceFunctions import sqexp

class ObsData(object):
    "class representing Observational Data and discrepancy between model and observations"
    def __init__(self, coords, data, unc, cov=sqexp, comm=COMM_WORLD):
        "create a new ObsData object given data and a covariance function"

        if not isinstance(comm, type(COMM_WORLD)):
            raise TypeError("comm must be an MPI communicator")

        self.comm = comm

        coords = np.array(coords, dtype=np.float64)

        if coords.ndim == 1:
            coords = np.reshape(coords, (-1, 1))

        assert coords.ndim == 2, "coords must be a 1D or 2D array"

        self.n_obs = coords.shape[0]
        self.n_dim = coords.shape[1]
        self.coords = np.copy(coords)

        data = np.array(data, dtype=np.float64)
        assert data.shape == (self.n_obs,), "data must be a 1D array with the same length as coords"
        if self.comm.rank == 0:
            self.data = np.copy(data)
        else:
            self.data = np.zeros(0)

        unc = np.array(unc, dtype=np.float64)
        unc = np.nan_to_num(unc)
        assert unc.shape == (self.n_obs,) or unc.shape == (), "bad shape for unc, must be an array or float"
        assert np.all(unc >= 0.), "all uncertainties must be non-negative"
        self.unc = np.copy(unc)

        self.cov = cov

    def get_n_dim(self):
        "returns number of dimensions in FEM model"

        return self.n_dim

    def get_n_obs(self):
        "returns number of observations"

        return self.n_obs

    def get_coords(self):
        "returns coordinate points as a numpy array"

        return self.coords

    def get_data(self):
        "returns observations or an empty array, depending on rank"

        return self.data

    def get_unc(self):
        "returns uncertainties (as standard deviation)"

        return self.unc

    def calc_K(self, params):
        "returns model discrepancy covariance matrix"

        params = np.array(params)
        assert params.shape == (2,), "parameters must have length 2"
        assert np.all(params > 0.), "parameters must all be positive"

        sigma = params[0]
        l = params[1]

        if self.comm.rank == 0:
            K = self.cov(self.coords, self.coords, sigma, l)
        else:
            K = np.zeros((0,0))

        return K

    def calc_K_plus_sigma(self, params):
        "return model discrepancy covariance plus observational data error"

        if self.comm.rank == 0:
            if self.unc.shape == ():
                sigma_dat = np.eye(self.n_obs)*self.unc**2
            else:
                sigma_dat = np.diag(self.unc**2)
        else:
            sigma_dat = np.zeros((0, 0))

        return self.calc_K(params) + sigma_dat
