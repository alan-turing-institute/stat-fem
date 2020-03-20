import numpy as np
from scipy.spatial.distance import cdist

def sqexp(x1, x2, sigma, l):
    """
    Squared Exponential Covariance Function

    Returns squared exponential covariance function computed at all pairs of values in coordinate
    values ``x1`` and ``x2``. This returns a 2D array with the first axis of the same length as
    the first axis of ``x1``, and the second axis of the same length as the first axis of ``x2``.
    If ``x1`` or ``x2`` are 1D, it broadcasts to 2D assuming that the first axis has length 1.

    Note parameters are assumed to be on log scale. ``sigma`` is the overall covariance scale,
    where ``exp(sigma)`` is the standard deviation, and ``l`` determines the correlation length,
    where ``exp(l)`` is the length scale.

    :param x1: first set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x1: ndarray
    :param x2: second set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x2: ndarray
    :param sigma: Covariance parameter on a logarithmic scale (``exp(sigma)`` gives the standard
                  deviation).
    :type sigma: float
    :param l: Correlation length on a logarithmic scale (``exp(l)`` gives the standard deviation).
    :type l: float
    :returns: Squared Exponential Covariance Matrix, a numpy array with shape ``(x1.shape[0], x2.shape[1])``
    :rtype: ndarray
    """

    x1 = np.array(x1)
    x2 = np.array(x2)

    r2 = calc_r2(x1, x2)

    return np.exp(2.*sigma)*np.exp(-0.5*r2*np.exp(-2.*l))

def sqexp_deriv(x1, x2, sigma, l):
    """
    Squared Exponential Covariance Function Derivative

    Returns the gradient of the squared exponential covariance function computed at all pairs of
    values in coordinate values ``x1`` and ``x2``. This returns a 3D array with the first axis of
    length 2 (representing the two derivative components), the second axis is of the same length as
    the first axis of ``x1``, and the third axis of the same length as the first axis of ``x2``.
    If ``x1`` or ``x2`` are 1D, it broadcasts to 2D assuming that the first axis has length 1.

    Note parameters are assumed to be on log scale. ``sigma`` is the overall covariance scale,
    where ``exp(sigma)`` is the standard deviation, and ``l`` determines the correlation length,
    where ``exp(l)`` is the length scale.

    :param x1: first set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x1: ndarray
    :param x2: second set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x2: ndarray
    :param sigma: Covariance parameter on a logarithmic scale (``exp(sigma)`` gives the standard
                  deviation).
    :type sigma: float
    :param l: Correlation length on a logarithmic scale (``exp(l)`` gives the standard deviation).
    :type l: float
    :returns: Squared Exponential Covariance Matrix gradient, a numpy array with shape
              ``(2, x1.shape[0], x2.shape[1])``
    :rtype: ndarray
    """

    x1 = np.array(x1)
    x2 = np.array(x2)

    deriv = np.zeros((2, x1.shape[0], x2.shape[0]))

    deriv[0] = 2.*sqexp(x1, x2, sigma, l)
    deriv[1] = sqexp(x1, x2, sigma, l)*calc_r2(x1, x2)*np.exp(-2.*l)

    return deriv

def sqexp_hessian(x1, x2, sigma, l):
    """
    Squared Exponential Covariance Function Hessian

    Returns the Hessian of the squared exponential covariance function computed at all pairs of
    values in coordinate values ``x1`` and ``x2``. This returns a 4D array with the first two axes of
    length 2 (representing the two derivative components), the third axis is of the same length as
    the first axis of ``x1``, and the fourth axis of the same length as the first axis of ``x2``.
    If ``x1`` or ``x2`` are 1D, it broadcasts to 2D assuming that the first axis has length 1.

    Note parameters are assumed to be on log scale. ``sigma`` is the overall covariance scale,
    where ``exp(sigma)`` is the standard deviation, and ``l`` determines the correlation length,
    where ``exp(l)`` is the length scale.

    :param x1: first set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x1: ndarray
    :param x2: second set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x2: ndarray
    :param sigma: Covariance parameter on a logarithmic scale (``exp(sigma)`` gives the standard
                  deviation).
    :type sigma: float
    :param l: Correlation length on a logarithmic scale (``exp(l)`` gives the standard deviation).
    :type l: float
    :returns: Squared Exponential Covariance Matrix Hessian, a numpy array with shape
              ``(2, 2, x1.shape[0], x2.shape[1])``
    :rtype: ndarray
    """

    x1 = np.array(x1)
    x2 = np.array(x2)

    hess = np.zeros((2, 2, x1.shape[0], x2.shape[0]))

    hess[0, 0] = 4.*sqexp(x1, x2, sigma, l)
    hess[0, 1] = 2.*sqexp(x1, x2, sigma, l)*calc_r2(x1, x2)*np.exp(-2.*l)
    hess[1, 0] = np.copy(hess[0, 1])
    hess[1, 1] = sqexp(x1, x2, sigma, l)*((calc_r2(x1, x2)*np.exp(-2.*l))**2 -
                                          2.*calc_r2(x1, x2)*np.exp(-2.*l))

    return hess

def calc_r2(x1, x2):
    """
    Compute Squared Distance between all pairs of points in two arrays

    Wrapper to the Scipy ``cdist`` function squared for computing squared distances
    between pairs of points in two arrays. This returns a 2D array with the first axis
    of the same length as the first axis of ``x1``, and the second axis of the same length
    as the first axis of ``x2``. If ``x1`` or ``x2`` are 1D, it broadcasts to 2D assuming
    that the first axis has length 1.

    :param x1: first set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x1: ndarray
    :param x2: second set of input coordinates. Must be a 1D or 2D numpy array (if 1D, it assumes the
               first axis has length 1).
    :type x2: ndarray
    :returns: Squared distance matrix, a numpy array with shape ``(x1.shape[0], x2.shape[1])``
    :rtype: ndarray
    """

    x1 = np.array(x1)
    x2 = np.array(x2)

    return cdist(np.atleast_2d(x1), np.atleast_2d(x2))**2