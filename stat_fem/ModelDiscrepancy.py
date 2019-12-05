import numpy as np
from firedrake import COMM_WORLD
from .CovarianceFunctions import sqexp
from .ObsData import ObsData

class ModelDiscrepancy(object):
    "class representing model discrepancy for a model and observations"
    def __init__(self, data_obs, rho, sigma, l, cov=sqexp):
        "create a new model discrepancy object given data and a covariance function"

        assert isinstance(data_obs, ObsData), "data_obs must be of type ObsData"
        assert rho > 0., "scaling factor must be positive"
        assert sigma > 0., "covariance scale must be positive"
        assert l > 0., "covariance length scale must be positive"

        # dataspace arrays only live on the root process, allocate dummy arrays for others

        if COMM_WORLD.rank == 0:
            self.K = cov(data_obs.get_coords(), data_obs.get_coords(), sigma, l)
            self.n_local = data_obs.n_obs
        else:
            self.K = np.zeros((0,0))
            self.n_local = 0
        self.sigma_dat = data_obs.get_unc()
        self.n_obs = data_obs.n_obs
        self.rho = float(rho)

    def get_rho(self):
        "returns scaling factor"

        return self.rho

    def get_K(self):
        "returns model discrepancy covariance matrix"

        return self.K

    def get_K_plus_sigma(self):
        "return model discrepancy covariance plus observational data error"

        if self.sigma_dat.shape == ():
            sigma_dat = np.eye(self.n_local)*self.sigma_dat**2
        else:
            if COMM_WORLD.rank == 0:
                sigma_dat = np.diag(self.sigma_dat**2)
            else:
                sigma_dat = np.zeros((0, 0))

        return self.K + sigma_dat