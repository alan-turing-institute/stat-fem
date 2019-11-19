import numpy as np
from numpy.testing import assert_allclose
import pytest
from ..CovarianceFunctions import sqexp

def test_sqexp():
    "test the squared exponential covariance function"

    # 1d input array

    x = np.array([[1.], [2.], [3.]])
    sigma = 1.
    l = 1.

    r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])

    f_expected = sigma**2*np.exp(-0.5*r**2/l**2)
    f_actual = sqexp(x, x, sigma, l)

    assert_allclose(f_expected, f_actual)

    # 2d input array

    x = np.array([[2., 1.], [0., 2.], [1., 1.]])
    sigma = 1.
    l = 1.

    r = np.array([[0.,          np.sqrt(5.), 1.         ],
                  [np.sqrt(5.), 0.,          np.sqrt(2.)],
                  [1.,          np.sqrt(2.), 0.         ]])

    f_expected = sigma**2*np.exp(-0.5*r**2/l**2)
    f_actual = sqexp(x, x, sigma, l)

    assert_allclose(f_expected, f_actual)

    # change parameters

    x = np.array([[1.], [2.], [3.]])
    sigma = 0.1
    l = 2.

    r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])

    f_expected = sigma**2*np.exp(-0.5*r**2/l**2)
    f_actual = sqexp(x, x, sigma, l)

    assert_allclose(f_expected, f_actual)

    # check failures for negative parameters

    sigma = -1.
    l = 1.

    with pytest.raises(AssertionError):
        sqexp(x, x, sigma, l)

    sigma = 1.
    l = -1.

    with pytest.raises(AssertionError):
        sqexp(x, x, sigma, l)

