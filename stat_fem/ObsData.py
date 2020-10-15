import numpy as np
from .covariance_functions import sqexp, sqexp_deriv

class ObsData(object):
    """
    Class representing Observational Data and discrepancy between model and observations

    This class holds information on observational data and uncertainties. It also serves
    as a wrapper to compute the model discrepancy (since the data coordinates and
    uncertainties are stored here). At the moment, it only implements the squared
    exponential kernel and assumes that data is not time varying.

    :ivar coords: Array holding coordinate locations of measurements. Must be 1D or 2D
                  where the first axis represents the different coordinate points and
                  the second axis represents the cartesian axis directions. If 1D,
                  assumes the second axis has length 1.
    :type coords: ndarray
    :ivar data: Sensor measurements at all of the given coordinate locations. Length
                must be the same as the first axis of ``coords``.
    :type data: ndarray
    :ivar unc: Uncertainty on data measurements as a standard deviation. Can be a single
               float, or an array of floats with the same length as the number of
               sensors. Must be non-negative.
    :type unc: float or ndarray
    """
    def __init__(self, coords, data, unc):
        """
        Create a new ObsData object

        Creates a new ``ObsData`` object given coordinates, data values, and uncertainties.
        Performs re-shaping of inputs and some checks on values.

        :param coords: Array holding coordinate locations of measurements. Must be 1D or 2D
                       where the first axis represents the different coordinate points and
                       the second axis represents the cartesian axis directions. If 1D,
                       assumes the second axis has length 1.
        :type coords: ndarray
        :param data: Sensor measurements at all of the given coordinate locations. Length
                     must be the same as the first axis of ``coords``.
        :type data: ndarray
        :param unc: Uncertainty on data measurements as a standard deviation. Can be a single
                    float, or an array of floats with the same length as the number of
                    sensors. Must be non-negative.
        :type unc: float or ndarray
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
        """
        Returns number of dimensions in FEM model

        Returns the number of spatial dimensions in FEM model as an integer.

        :returns: Number of spatial dimensions
        :rtype: int
        """

        return self.n_dim

    def get_n_obs(self):
        """
        Returns number of observations

        Returns the number of sensor observations as an integer.

        :returns: Number of sensor measurements
        :rtype: int
        """

        return self.n_obs

    def get_coords(self):
        """
        Returns coordinate points as a numpy array

        Returns coordinate points where sensor measurements have been made as a numpy array.
        2D array with the first axis representing the different sensors, and the second
        axis represents the different spatial dimensions.

        :returns: Coordinate array, a 2D numpy array
        :rtype: ndarray
        """

        return self.coords

    def get_data(self):
        """
        Returns sensor observations as a numpy array

        Returns sensor observations as a 1D numpy array. Length is the same as the number of sensors.

        :returns: Coordinate array, a 1D numpy array
        :rtype: ndarray
        """

        return self.data

    def get_unc(self):
        """
        Returns uncertainties

        Returns data measurement uncertainty as a standard deviation, either a float or a numpy array
        if uncertainties differ across sensors.

        :returns: Uncertainty as a standard deviation, either a float or a numpy array
        :rtype: float or ndarray
        """

        return self.unc

    def calc_K(self, params):
        """
        Returns model discrepancy covariance matrix

        Computes the model discrepancy covariance matrix for the given parameters. Assumes that
        the discrepancy is a multivariate normal distribution with zero mean and a squared
        exponential covariance matrix. Params are given on a logarithmic scale to enforce
        positivity, the first parameter is the overall covariance scale (actual covariance is
        found by taking exp(2.*params[0]) and the second is the spatial correlation length scale
        (actual correlation length is exp(params[1])). Returns a numpy array with shape
        ``(n_obs, n_obs)``.

        :param params: Covariance function parameters on a logarithmic scale. Must be a numpy
                       array of length 2 (first parameter is the overall covariance scale, second
                       determines the correlation length scale).
        :type params: ndarray
        :returns: Model discrepancy covariance matrix, shape is ``(n_obs, n_obs)``.
        :rtype: ndarray
        """

        params = np.array(params)
        assert params.shape == (2,), "parameters must have length 2"

        sigma = params[0]
        l = params[1]

        return sqexp(self.coords, self.coords, sigma, l)

    def calc_K_plus_sigma(self, params):
        """
        Returns model discrepancy covariance matrix plus observation error

        Computes the model discrepancy covariance matrix for the given parameters plus the
        observational error. Assumes that the discrepancy is a multivariate normal distribution
        with zero mean and a squared exponential covariance matrix. Params are given on a
        logarithmic scale to enforce positivity, the first parameter is the overall covariance
        scale (actual covariance is found by taking exp(2.*params[0]) and the second is the
        spatial correlation length scale (actual correlation length is exp(params[1])).
        Returns a numpy array with shape ``(n_obs, n_obs)``.

        :param params: Covariance function parameters on a logarithmic scale. Must be a numpy
                       array of length 2 (first parameter is the overall covariance scale, second
                       determines the correlation length scale).
        :type params: ndarray
        :returns: Model discrepancy covariance matrix plus observational error, shape is
                  ``(n_obs, n_obs)``.
        :rtype: ndarray
        """

        if self.unc.shape == ():
            sigma_dat = np.eye(self.n_obs)*self.unc**2
        else:
            sigma_dat = np.diag(self.unc**2)

        return self.calc_K(params) + sigma_dat

    def calc_K_deriv(self, params):
        """
        Returns derivative of model discrepancy covariance matrix wrt parameters

        Computes the derivative of the model discrepancy covariance matrix with respect to the
        input parameters. Assumes that the discrepancy is a multivariate normal distribution
        with zero mean and a squared exponential covariance matrix. Params are given on a
        logarithmic scale to enforce positivity, the first parameter is the overall covariance
        scale (actual covariance is found by taking exp(2.*params[0]) and the second is the
        spatial correlation length scale (actual correlation length is exp(params[1])).
        Returns a numpy array with shape ``(2, n_obs, n_obs)``.

        :param params: Covariance function parameters on a logarithmic scale. Must be a numpy
                       array of length 2 (first parameter is the overall covariance scale, second
                       determines the correlation length scale).
        :type params: ndarray
        :returns: Model discrepancy covariance matrix derivative with respect to the parameters,
                  shape is ``(2, n_obs, n_obs)``.
        :rtype: ndarray
        """

        params = np.array(params)
        assert params.shape == (2,), "parameters must have length 2"

        sigma = params[0]
        l = params[1]

        return sqexp_deriv(self.coords, self.coords, sigma, l)
