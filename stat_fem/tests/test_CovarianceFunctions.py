# import numpy as np
# from numpy.testing import assert_allclose
# import pytest
# from ..CovarianceFunctions import sqexp, sqexp_deriv, sqexp_hessian, calc_r2
#
# def test_sqexp():
#     "test the squared exponential covariance function"
#
#     # 1d input array
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = 0.
#     l = 0.
#
#     r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
#
#     f_expected = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_actual = sqexp(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, atol=1.e-10)
#
#     # 2d input array
#
#     x = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     sigma = 0.
#     l = 0.
#
#     r = np.array([[0.,          np.sqrt(5.), 1.         ],
#                   [np.sqrt(5.), 0.,          np.sqrt(2.)],
#                   [1.,          np.sqrt(2.), 0.         ]])
#
#     f_expected = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_actual = sqexp(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, atol=1.e-10)
#
#     # change parameters
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = np.log(0.1)
#     l = np.log(2.)
#
#     r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
#
#     f_expected = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_actual = sqexp(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, atol=1.e-10)
#
# def test_sqexp_deriv():
#     "test derivative of squared exponential"
#
#     # 1d input array
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = 0.
#     l = 0.
#
#     r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = 2.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_expected[1] = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*r**2/np.exp(l)**2
#     f_actual = sqexp_deriv(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, atol=1.e-10)
#
#     # 2d input array
#
#     x = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     sigma = 0.
#     l = 0.
#
#     r = np.array([[0.,          np.sqrt(5.), 1.         ],
#                   [np.sqrt(5.), 0.,          np.sqrt(2.)],
#                   [1.,          np.sqrt(2.), 0.         ]])
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = 2.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_expected[1] = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*r**2/np.exp(l)**2
#     f_actual = sqexp_deriv(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, atol=1.e-10)
#
#     # change parameters
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = np.log(0.1)
#     l = np.log(2.)
#
#     r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = 2.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_expected[1] = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*r**2/np.exp(l)**2
#     f_actual = sqexp_deriv(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, atol=1.e-10)
#
# def test_sqexp_deriv_fd():
#     "test sqexp_deriv using finite differences"
#
#     dx = 1.e-6
#
#     # 1d input array
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = 0.
#     l = 0.
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = (sqexp(x, x, sigma + dx, l) - sqexp(x, x, sigma, l))/dx
#     f_expected[1] = (sqexp(x, x, sigma, l + dx) - sqexp(x, x, sigma, l))/dx
#
#     f_actual = sqexp_deriv(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, rtol=1.e-5, atol=1.e-10)
#
#     # 2d input array
#
#     x = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     sigma = 0.
#     l = 0.
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = (sqexp(x, x, sigma + dx, l) - sqexp(x, x, sigma, l))/dx
#     f_expected[1] = (sqexp(x, x, sigma, l + dx) - sqexp(x, x, sigma, l))/dx
#     f_actual = sqexp_deriv(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, rtol=1.e-5, atol=1.e-10)
#
#     # change parameters
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = np.log(0.1)
#     l = np.log(2.)
#
#     r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = (sqexp(x, x, sigma + dx, l) - sqexp(x, x, sigma, l))/dx
#     f_expected[1] = (sqexp(x, x, sigma, l + dx) - sqexp(x, x, sigma, l))/dx
#     f_actual = sqexp_deriv(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, rtol=1.e-5, atol=1.e-10)
#
# def test_sqexp_hessian():
#     "test derivative of squared exponential"
#
#     # 1d input array
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = 0.
#     l = 0.
#
#     r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
#
#     f_expected = np.zeros((2, 2, 3, 3))
#     f_expected[0, 0] = 4.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_expected[0, 1] = 2.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*r**2/np.exp(l)**2
#     f_expected[1, 0] = f_expected[0, 1]
#     f_expected[1, 1] = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*(r**4/np.exp(l)**4 -
#                                                                         2.*r**2*np.exp(l)**2)
#     f_actual = sqexp_hessian(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, atol=1.e-10)
#
#     # 2d input array
#
#     x = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     sigma = 0.
#     l = 0.
#
#     r = np.array([[0.,          np.sqrt(5.), 1.         ],
#                   [np.sqrt(5.), 0.,          np.sqrt(2.)],
#                   [1.,          np.sqrt(2.), 0.         ]])
#
#     f_expected = np.zeros((2, 2, 3, 3))
#     f_expected[0, 0] = 4.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_expected[0, 1] = 2.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*r**2/np.exp(l)**2
#     f_expected[1, 0] = f_expected[0, 1]
#     f_expected[1, 1] = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*(r**4/np.exp(l)**4 -
#                                                                         2.*r**2*np.exp(l)**2)
#     f_actual = sqexp_hessian(x, x, sigma, l)
#
#     # change parameters
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = np.log(0.1)
#     l = np.log(2.)
#
#     r = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
#
#     f_expected = np.zeros((2, 2, 3, 3))
#     f_expected[0, 0] = 4.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_expected[0, 1] = 2.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*r**2/np.exp(l)**2
#     f_expected[1, 0] = f_expected[0, 1]
#     f_expected[1, 1] = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*(r**4/np.exp(l)**4 -
#                                                                         2.*r**2*np.exp(l)**2)
#     f_actual = sqexp_hessian(x, x, sigma, l)
#
# def test_sqexp_hessian_fd():
#     "test the hessian function using finite differences"
#
#     dx = 1.e-6
#
#     # 1d input array
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = 0.
#     l = 0.
#
#     f_expected = np.zeros((2, 2, 3, 3))
#     f_expected[0, 0] = (sqexp_deriv(x, x, sigma + dx, l)[0] - sqexp_deriv(x, x, sigma, l)[0])/dx
#     f_expected[0, 1] = (sqexp_deriv(x, x, sigma, l + dx)[0] - sqexp_deriv(x, x, sigma, l)[0])/dx
#     f_expected[1, 0] = (sqexp_deriv(x, x, sigma + dx, l)[1] - sqexp_deriv(x, x, sigma, l)[1])/dx
#     f_expected[1, 1] = (sqexp_deriv(x, x, sigma, l + dx)[1] - sqexp_deriv(x, x, sigma, l)[1])/dx
#
#     f_actual = sqexp_hessian(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, rtol=1.e-5, atol=1.e-5)
#
#     # 2d input array
#
#     x = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     sigma = 0.
#     l = 0.
#
#     f_expected = np.zeros((2, 2, 3, 3))
#     f_expected[0, 0] = (sqexp_deriv(x, x, sigma + dx, l)[0] - sqexp_deriv(x, x, sigma, l)[0])/dx
#     f_expected[0, 1] = (sqexp_deriv(x, x, sigma, l + dx)[0] - sqexp_deriv(x, x, sigma, l)[0])/dx
#     f_expected[1, 0] = (sqexp_deriv(x, x, sigma + dx, l)[1] - sqexp_deriv(x, x, sigma, l)[1])/dx
#     f_expected[1, 1] = (sqexp_deriv(x, x, sigma, l + dx)[1] - sqexp_deriv(x, x, sigma, l)[1])/dx
#
#     f_actual = sqexp_hessian(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, rtol=1.e-5, atol=1.e-5)
#
#     # change parameters
#
#     x = np.array([[1.], [2.], [3.]])
#     sigma = np.log(0.1)
#     l = np.log(2.)
#
#     f_expected = np.zeros((2, 2, 3, 3))
#     f_expected[0, 0] = (sqexp_deriv(x, x, sigma + dx, l)[0] - sqexp_deriv(x, x, sigma, l)[0])/dx
#     f_expected[0, 1] = (sqexp_deriv(x, x, sigma, l + dx)[0] - sqexp_deriv(x, x, sigma, l)[0])/dx
#     f_expected[1, 0] = (sqexp_deriv(x, x, sigma + dx, l)[1] - sqexp_deriv(x, x, sigma, l)[1])/dx
#     f_expected[1, 1] = (sqexp_deriv(x, x, sigma, l + dx)[1] - sqexp_deriv(x, x, sigma, l)[1])/dx
#
#     f_actual = sqexp_hessian(x, x, sigma, l)
#
#     assert_allclose(f_expected, f_actual, rtol=1.e-5, atol=1.e-5)
#
# def test_calc_r2():
#     "test the method to calculate r2"
#
#     # 1d input array
#
#     x = np.array([[1.], [2.], [3.]])
#
#     r_expected = np.array([[0., 1., 4.], [1., 0., 1.], [4., 1., 0.]])
#     r_actual = calc_r2(x, x)
#
#     assert_allclose(r_expected, r_actual)
#
#     # 2d input array
#
#     x = np.array([[2., 1.], [0., 2.], [1., 1.]])
#
#     r_expected = np.array([[0., 5., 1.],
#                   [5., 0., 2.],
#                   [1., 2., 0.]])
#     r_actual = calc_r2(x, x)
#
#     assert_allclose(r_expected, r_actual)
