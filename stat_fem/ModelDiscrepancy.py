import numpy as np
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

        self.rho = float(rho)
        self.K = cov(data_obs.get_coords(), data_obs.get_coords(), sigma, l)
        self.sigma_dat = data_obs.get_unc()
        self.n_obs = data_obs.n_obs

    def get_rho(self):
        "returns scaling factor"

        return self.rho

    def get_K(self):
        "returns model discrepancy covariance matrix"

        return self.K

    def get_K_plus_sigma(self):
        "compute cholesky factorization if needed"

        if self.sigma_dat.shape == ():
            sigma_dat = np.eye(self.n_obs)*self.sigma_dat**2
        else:
            sigma_dat = np.diag(self.sigma_dat**2)

        return self.K + sigma_dat