# import numpy as np
# from numpy.testing import assert_allclose
# import pytest
# from firedrake import COMM_WORLD
# from ..ObsData import ObsData
#
# def test_ObsData_init():
#     "test ObsData init method"
#
#     # simple 1D coords and single uncertainty
#
#     coords = np.array([1., 2., 3.])
#     data = np.array([2., 3., 4.])
#     unc = 1.
#
#     od = ObsData(coords, data, unc)
#
#     assert_allclose(od.coords, np.reshape(coords, (-1, 1)))
#     assert_allclose(od.data, data)
#     assert_allclose(od.unc, unc)
#     assert od.n_dim == 1
#     assert od.n_obs == 3
#
#     # 2D coords and array uncertainties
#
#     coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
#     unc = np.array([4., 5., 6.])
#
#     od = ObsData(coords, data, unc)
#
#     assert_allclose(od.coords, coords)
#     assert_allclose(od.data, data)
#     assert_allclose(od.unc, unc)
#     assert od.n_dim == 2
#     assert od.n_obs == 3
#
#     # unc is none
#
#     unc = None
#
#     od = ObsData(coords, data, unc)
#
#     assert_allclose(od.coords, coords)
#     assert_allclose(od.data, data)
#     assert_allclose(od.unc, 0.)
#     assert od.n_dim == 2
#     assert od.n_obs == 3
#
# def test_ObsData_init_failures():
#     "check situations where ObsData should fail"
#
#     # bad shape for coords
#
#     coords = np.array([[[1., 2., 3.]]])
#     data = np.array([3., 4.])
#     unc = 1.
#
#     with pytest.raises(AssertionError):
#         od = ObsData(coords, data, unc)
#
#     # shape of coords and data don't agree
#
#     coords = np.array([1., 2., 3.])
#     data = np.array([3., 4.])
#     unc = 1.
#
#     with pytest.raises(AssertionError):
#         od = ObsData(coords, data, unc)
#
#     # bad shape for unc
#
#     coords = np.array([1., 2., 3.])
#     data = np.array([2., 3., 4.])
#     unc = np.array([1., 2.])
#
#     with pytest.raises(AssertionError):
#         od = ObsData(coords, data, unc)
#
#     # negative uncertainty
#
#     coords = np.array([1., 2., 3.])
#     data = np.array([2., 3., 4.])
#     unc = -1.
#
#     with pytest.raises(AssertionError):
#         od = ObsData(coords, data, unc)
#
# def test_ObsData_calc_K():
#     "test the calc_K method of ObsData"
#
#     coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     data = np.array([1., 2., 3.])
#     unc = 0.1
#
#     params = np.log(np.ones(2))
#     sigma = params[0]
#     l = params[1]
#
#     od = ObsData(coords, data, unc)
#
#     r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
#     K = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#
#     assert_allclose(od.calc_K(params), K)
#
#     # fails if params has wrong length
#
#     with pytest.raises(AssertionError):
#         od.calc_K(np.ones(3))
#
# def test_ObsData_calc_K_plus_sigma():
#     "test the get_K method of ModelDiscrepancy"
#
#     coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     data = np.array([1., 2., 3.])
#     unc = 0.1
#
#     params = np.log(np.ones(2))
#     sigma = params[0]
#     l = params[1]
#
#     od = ObsData(coords, data, unc)
#
#     r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
#     K = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2) + np.eye(3)*unc**2
#
#     assert_allclose(od.calc_K_plus_sigma(params), K)
#
# def test_ObsData_calc_K_deriv():
#     "test the calc_K_deriv method of ObsData"
#
#     coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     data = np.array([1., 2., 3.])
#     unc = 0.1
#
#     params = np.log(np.ones(2))
#     sigma = params[0]
#     l = params[1]
#
#     od = ObsData(coords, data, unc)
#
#     r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = 2.*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)
#     f_expected[1] = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)*r**2/np.exp(l)**2
#
#     assert_allclose(od.calc_K_deriv(params), f_expected)
#
# def test_ObsData_calc_K_deriv_fd():
#     "test the calc_K_deriv method of ObsData with finite differences"
#
#     dx = 1.e-6
#
#     coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
#     data = np.array([1., 2., 3.])
#     unc = 0.1
#
#     params = np.log(np.ones(2))
#     sigma = params[0]
#     l = params[1]
#
#     od = ObsData(coords, data, unc)
#
#     f_expected = np.zeros((2, 3, 3))
#     f_expected[0] = (od.calc_K(params + np.array([dx, 0.])) - od.calc_K(params))/dx
#     f_expected[1] = (od.calc_K(params + np.array([0., dx])) - od.calc_K(params))/dx
#
#     assert_allclose(od.calc_K_deriv(params), f_expected, atol=1.e-5)
#
# def test_ObsData_get_n_dim():
#     "test the get_n_dim method of ObsData"
#
#     coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
#     data = np.array([1., 2., 3.])
#     unc = np.array([4., 5., 6.])
#
#     od = ObsData(coords, data, unc)
#
#     assert od.get_n_dim() == 2
#
# def test_ObsData_get_n_obs():
#     "test the get_n_obs method of ObsData"
#
#     coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
#     data = np.array([1., 2., 3.])
#     unc = np.array([4., 5., 6.])
#
#     od = ObsData(coords, data, unc)
#
#     assert od.get_n_obs() == 3
#
# def test_ObsData_get_coords():
#     "test the get_coords method of ObsData"
#
#     coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
#     data = np.array([1., 2., 3.])
#     unc = np.array([4., 5., 6.])
#
#     od = ObsData(coords, data, unc)
#
#     assert_allclose(od.get_coords(), coords)
#
# def test_ObsData_get_data():
#     "test the get_data method of ObsData"
#
#     coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
#     data = np.array([1., 2., 3.])
#     unc = np.array([4., 5., 6.])
#
#     od = ObsData(coords, data, unc)
#
#     assert_allclose(od.get_data(), data)
#
# def test_ObsData_get_unc():
#     "test the get_unc method of ObsData"
#
#     coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
#     data = np.array([1., 2., 3.])
#     unc = np.array([4., 5., 6.])
#
#     od = ObsData(coords, data, unc)
#
#     assert_allclose(od.get_unc(), unc)
#
#
