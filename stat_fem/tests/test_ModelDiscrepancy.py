import numpy as np
from numpy.testing import assert_allclose
import pytest
from ..ModelDiscrepancy import ModelDiscrepancy
from ..ObsData import ObsData

def test_ModelDiscrepancy_init():
    "test ModelDiscrepancy"

    coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
    data = np.array([1., 2., 3.])
    unc = 0.1

    sigma = 1.
    l = 1.

    od = ObsData(coords, data, unc)

    md = ModelDiscrepancy(od, sigma, l)

    assert md.n_obs == 3
    assert_allclose(md.sigma_dat, unc)

    r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
    K = sigma**2*np.exp(-0.5*r**2/l**2)

    assert_allclose(md.K, K)

def test_ModelDiscrepancy_get_K():
    "test the get_K method of ModelDiscrepancy"

    coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
    data = np.array([1., 2., 3.])
    unc = 0.1

    sigma = 1.
    l = 1.

    od = ObsData(coords, data, unc)

    md = ModelDiscrepancy(od, sigma, l)

    r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
    K = sigma**2*np.exp(-0.5*r**2/l**2)

    assert_allclose(md.get_K(), K)

def test_ModelDiscrepancy_get_K_plus_sigma():
    "test the get_K method of ModelDiscrepancy"

    coords = np.array([[2., 1.], [0., 2.], [1., 1.]])
    data = np.array([1., 2., 3.])
    unc = 0.1

    sigma = 1.
    l = 1.

    od = ObsData(coords, data, unc)

    md = ModelDiscrepancy(od, sigma, l)

    r = np.array([[0., np.sqrt(5.), 1.], [np.sqrt(5.), 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])
    K = sigma**2*np.exp(-0.5*r**2/l**2) + np.eye(3)*unc**2

    assert_allclose(md.get_K_plus_sigma(), K)