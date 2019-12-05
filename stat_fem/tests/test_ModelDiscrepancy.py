import numpy as np
from numpy.testing import assert_allclose
import pytest
from firedrake import COMM_WORLD
from ..ModelDiscrepancy import ModelDiscrepancy
from ..ObsData import ObsData

def test_ModelDiscrepancy_init():
    "test ModelDiscrepancy"

    coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
    data = np.array([1., 2., 3.])
    unc = 0.1

    rho = 1.
    sigma = 1.
    l = 1.

    od = ObsData(coords, data, unc)

    md = ModelDiscrepancy(od, rho, sigma, l)

    assert md.n_obs == 3
    assert_allclose(md.rho, rho)
    assert_allclose(md.sigma_dat, unc)

    r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
    K = sigma**2*np.exp(-0.5*r**2/l**2)

    if COMM_WORLD.rank == 0:
        assert_allclose(md.K, K)
        assert md.n_local == md.n_obs
    else:
        assert md.K.shape == (0, 0)

def test_ModelDiscrepancy_get_K():
    "test the get_K method of ModelDiscrepancy"

    coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
    data = np.array([1., 2., 3.])
    unc = 0.1

    rho = 1.
    sigma = 1.
    l = 1.

    od = ObsData(coords, data, unc)

    md = ModelDiscrepancy(od, rho, sigma, l)

    if COMM_WORLD.rank == 0:
        r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
        K = sigma**2*np.exp(-0.5*r**2/l**2)
    else:
        K = np.zeros((0,0))

    assert_allclose(md.get_K(), K)

def test_ModelDiscrepancy_get_K_plus_sigma():
    "test the get_K method of ModelDiscrepancy"

    coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
    data = np.array([1., 2., 3.])
    unc = 0.1

    rho = 1.
    sigma = 1.
    l = 1.

    od = ObsData(coords, data, unc)

    md = ModelDiscrepancy(od, rho, sigma, l)

    if COMM_WORLD.rank == 0:
        r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
        K = sigma**2*np.exp(-0.5*r**2/l**2) + np.eye(3)*unc**2
    else:
        K = np.zeros((0,0))

    assert_allclose(md.get_K_plus_sigma(), K)

def test_ModelDiscrepancy_get_rho():
    "test the get_rho method of ModelDiscrepancy"

    coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
    data = np.array([1., 2., 3.])
    unc = 0.1

    rho = 1.
    sigma = 1.
    l = 1.

    od = ObsData(coords, data, unc)

    md = ModelDiscrepancy(od, rho, sigma, l)

    assert_allclose(md.get_rho(), rho)