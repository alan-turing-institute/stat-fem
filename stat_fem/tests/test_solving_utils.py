import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from firedrake.function import Function
from firedrake import COMM_WORLD
import pytest
from ..solving_utils import solve_forcing_covariance, interp_covariance_to_data
from ..InterpolationMatrix import InterpolationMatrix
from .helper_funcs import nx, my_ensemble, comm, mesh, fs, A, b, fc, meshcoords, coords, interp
from .helper_funcs import A_numpy, cov

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_solve_forcing_covariance(comm, fs, A, b, fc, A_numpy, cov):
    "test solve_forcing_covariance"

    rhs = Function(fs).vector()
    rhs.set_local(np.ones(fc.get_nx_local()))

    result = solve_forcing_covariance(fc, A, rhs)

    result_actual = result.gather()

    result_expected = np.linalg.solve(A_numpy, np.ones(nx + 1))
    result_expected = np.dot(cov, result_expected)
    result_expected = np.linalg.solve(A_numpy, result_expected)

    assert_allclose(result_expected, result_actual, atol = 1.e-10)

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
def test_solve_forcing_covariance_parallel(my_ensemble, comm, fs, A, b, fc, A_numpy, cov):
    "test that solve_forcing_covariance can be called independently from an ensemble process"

    if my_ensemble.ensemble_comm.rank == 0:

        rhs = Function(fs).vector()
        rhs.set_local(np.ones(fc.get_nx_local()))

        result = solve_forcing_covariance(fc, A, rhs)

        result_actual = result.gather()

        result_expected = np.linalg.solve(A_numpy, np.ones(nx + 1))
        result_expected = np.dot(cov, result_expected)
        result_expected = np.linalg.solve(A_numpy, result_expected)

        assert_allclose(result_expected, result_actual, atol = 1.e-10)

    elif my_ensemble.ensemble_comm.rank == 1:

        rhs = Function(fs).vector()
        rhs.set_local(0.5*np.ones(fc.get_nx_local()))

        result = solve_forcing_covariance(fc, A, rhs)

        result_actual = result.gather()

        result_expected = np.linalg.solve(A_numpy, 0.5*np.ones(nx + 1))
        result_expected = np.dot(cov, result_expected)
        result_expected = np.linalg.solve(A_numpy, result_expected)

        assert_allclose(result_expected, result_actual, atol = 1.e-10)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix_interp_covariance_to_data(fs, A, fc, coords, interp, cov, A_numpy):
    "test the interp_covariance_to_data method"

    # simple 1D test

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)
    im.assemble()

    assert im.is_assembled

    result_expected = np.linalg.solve(A_numpy, interp)
    result_expected = np.dot(cov, result_expected)
    result_expected = np.linalg.solve(A_numpy, result_expected)
    result_expected = np.dot(interp.T, result_expected)

    result_actual = interp_covariance_to_data(im, fc, A, im)

    if COMM_WORLD.rank == 0:
        assert_allclose(result_expected, result_actual, atol=1.e-10)
    else:
        assert result_actual.shape == (0, 0)

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1, 2], indirect=["coords"])
def test_InterpolationMatrix_interp_covariance_to_data_ensemble(my_ensemble, fs, A, fc, coords,
                                                                interp, A_numpy, cov):
    "test the interp_covariance_to_data method"

    # simple 1D test

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)
    im.assemble()

    result_expected = np.linalg.solve(A_numpy, interp)
    result_expected = np.dot(cov, result_expected)
    result_expected = np.linalg.solve(A_numpy, result_expected)
    result_expected = np.dot(interp.T, result_expected)

    result_actual = interp_covariance_to_data(im, fc, A, im, my_ensemble.ensemble_comm)

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        assert_allclose(result_expected, result_actual, atol=1.e-10)
    else:
        assert result_actual.shape == (0, 0)