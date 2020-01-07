import numpy as np
from scipy.spatial.distance import cdist

def sqexp(x1, x2, sigma, l):
    "returns squared exponential covariance function. note parameters are assumed to be on log scale"

    x1 = np.array(x1)
    x2 = np.array(x2)

    r2 = calc_r2(x1, x2)

    return np.exp(2.*sigma)*np.exp(-0.5*r2*np.exp(-2.*l))

def sqexp_deriv(x1, x2, sigma, l):
    "returns partial derivatives of the squared exponential covariance function"

    x1 = np.array(x1)
    x2 = np.array(x2)

    deriv = np.zeros((2, x1.shape[0], x2.shape[0]))

    deriv[0] = 2.*sqexp(x1, x2, sigma, l)
    deriv[1] = sqexp(x1, x2, sigma, l)*calc_r2(x1, x2)*np.exp(-2.*l)

    return deriv

def sqexp_hessian(x1, x2, sigma, l):
    "returns hessian of the squared exponential covariance function"

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
    "compute squared distance between points"

    x1 = np.array(x1)
    x2 = np.array(x2)

    return cdist(np.atleast_2d(x1), np.atleast_2d(x2))**2