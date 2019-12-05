import numpy as np
from firedrake import COMM_WORLD

class ObsData(object):
    "represents a set of observations made at a set of coordinates with a corresponding uncertainty"
    def __init__(self, coords, data, unc):
        """
        create a new data object from locations, observations, and uncertainties

        coords is a 1D or 2D array holding the spatial locations of the measurements
             if 1D, then assumes that FEM is explicitly 1D in nature. If 2D, then
             FEM can be 1D, 2D, or 3D with the second index
        data is the measurements itself, a 1D numpy array with the same length as the
             first dimension of coords
        unc is the measurement error (as a standard deviation), assumed to be independent
             for each measurement can be either an array of the same length as data or
             a single float that is applied to all numbers. None or NaN is assumed to be zero
        """

        coords = np.array(coords, dtype=np.float64)

        if coords.ndim == 1:
            coords = np.reshape(coords, (-1, 1))

        assert coords.ndim == 2, "coords must be a 1D or 2D array"

        self.n_obs = coords.shape[0]
        self.n_dim = coords.shape[1]
        self.coords = np.copy(coords)

        data = np.array(data, dtype=np.float64)
        assert data.shape == (self.n_obs,), "data must be a 1D array with the same length as coords"
        self.data = np.copy(data)

        unc = np.array(unc, dtype=np.float64)
        unc = np.nan_to_num(unc)
        assert unc.shape == (self.n_obs,) or unc.shape == (), "bad shape for unc, must be an array or float"
        assert np.all(unc >= 0.), "all uncertainties must be non-negative"
        self.unc = np.copy(unc)

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
        "returns observations"

        if COMM_WORLD.rank == 0:
            return self.data
        else:
            return np.zeros(0)

    def get_unc(self):
        "returns uncertainties (as standard deviation)"

        return self.unc

    def __str__(self):
        "returns a string representation"

        outstr = ("Observational Data:\n" +
                  "Number of dimensions:\n" +
                  "{}\n".format(self.get_n_dim()) +
                  "Number of observations:\n" +
                  "{}\n".format(self.get_n_obs()) +
                  "Coordinates:\n" +
                  "{}\n".format(self.get_coords()) +
                  "Data:\n" +
                  "{}\n".format(self.data) +
                  "Uncertainty:\n" +
                  "{}".format(self.get_unc()))

        return outstr