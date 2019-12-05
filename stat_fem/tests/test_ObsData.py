import pytest
import numpy as np
from numpy.testing import assert_allclose
from firedrake import COMM_WORLD
from ..ObsData import ObsData

def test_ObsData_init():
    "test ObsData"

    # simple 1D coords and single uncertainty

    coords = np.array([1., 2., 3.])
    data = np.array([2., 3., 4.])
    unc = 1.

    od = ObsData(coords, data, unc)

    assert_allclose(od.coords, np.reshape(coords, (-1, 1)))
    assert_allclose(od.data, data)
    assert_allclose(od.unc, unc)
    assert od.n_dim == 1
    assert od.n_obs == 3

    # 2D coords and array uncertainties

    coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
    unc = np.array([4., 5., 6.])

    od = ObsData(coords, data, unc)

    assert_allclose(od.coords, coords)
    assert_allclose(od.data, data)
    assert_allclose(od.unc, unc)
    assert od.n_dim == 2
    assert od.n_obs == 3

    # unc is none

    unc = None

    od = ObsData(coords, data, unc)

    assert_allclose(od.coords, coords)
    assert_allclose(od.data, data)
    assert_allclose(od.unc, 0.)
    assert od.n_dim == 2
    assert od.n_obs == 3

def test_ObsData_init_failures():
    "check situations where ObsData should fail"

    # bad shape for coords

    coords = np.array([[[1., 2., 3.]]])
    data = np.array([3., 4.])
    unc = 1.

    with pytest.raises(AssertionError):
        od = ObsData(coords, data, unc)

    # shape of coords and data don't agree

    coords = np.array([1., 2., 3.])
    data = np.array([3., 4.])
    unc = 1.

    with pytest.raises(AssertionError):
        od = ObsData(coords, data, unc)

    # bad shape for unc

    coords = np.array([1., 2., 3.])
    data = np.array([2., 3., 4.])
    unc = np.array([1., 2.])

    with pytest.raises(AssertionError):
        od = ObsData(coords, data, unc)

    # negative uncertainty

    coords = np.array([1., 2., 3.])
    data = np.array([2., 3., 4.])
    unc = -1.

    with pytest.raises(AssertionError):
        od = ObsData(coords, data, unc)

def test_ObsData_get_n_dim():
    "test the get_n_dim method of ObsData"

    coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = np.array([1., 2., 3.])
    unc = np.array([4., 5., 6.])

    od = ObsData(coords, data, unc)

    assert od.get_n_dim() == 2

def test_ObsData_get_n_obs():
    "test the get_n_obs method of ObsData"

    coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = np.array([1., 2., 3.])
    unc = np.array([4., 5., 6.])

    od = ObsData(coords, data, unc)

    assert od.get_n_obs() == 3

def test_ObsData_get_coords():
    "test the get_coords method of ObsData"

    coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = np.array([1., 2., 3.])
    unc = np.array([4., 5., 6.])

    od = ObsData(coords, data, unc)

    assert_allclose(od.get_coords(), coords)

def test_ObsData_get_data():
    "test the get_data method of ObsData"

    coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = np.array([1., 2., 3.])
    unc = np.array([4., 5., 6.])

    od = ObsData(coords, data, unc)

    if COMM_WORLD.rank == 0:
        assert_allclose(od.get_data(), data)
    else:
        assert od.get_data().shape == (0,)

def test_ObsData_get_unc():
    "test the get_unc method of ObsData"

    coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = np.array([1., 2., 3.])
    unc = np.array([4., 5., 6.])

    od = ObsData(coords, data, unc)

    assert_allclose(od.get_unc(), unc)

def test_ObsData_str():
    "test the string method of ObsData"

    coords = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = np.array([1., 2., 3.])
    unc = np.array([4., 5., 6.])

    od = ObsData(coords, data, unc)

    outstr = ("Observational Data:\n" +
              "Number of dimensions:\n" +
              "{}\n".format(2) +
              "Number of observations:\n" +
              "{}\n".format(3) +
              "Coordinates:\n" +
              "{}\n".format(coords) +
              "Data:\n" +
              "{}\n".format(data) +
              "Uncertainty:\n" +
              "{}".format(unc))

    assert outstr == str(od)
