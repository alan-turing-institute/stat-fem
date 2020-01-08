import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from firedrake.function import Function
from firedrake import COMM_WORLD
import pytest
from ..solving_utils import _solve_forcing_covariance
from .helper_funcs import nx, my_ensemble, comm, mesh, fs, A, b, fc, meshcoords
from .helper_funcs import A_numpy, cov

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_solve_forcing_covariance(comm, fs, A, b, fc, A_numpy, cov):
    "test solve_forcing_covariance"

    rhs = Function(fs).vector()
    rhs.set_local(np.ones(fc.get_nx_local()))

    result = _solve_forcing_covariance(fc, A, rhs)

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

        result = _solve_forcing_covariance(fc, A, rhs)

        result_actual = result.gather()

        result_expected = np.linalg.solve(A_numpy, np.ones(nx + 1))
        result_expected = np.dot(cov, result_expected)
        result_expected = np.linalg.solve(A_numpy, result_expected)

        assert_allclose(result_expected, result_actual, atol = 1.e-10)

    elif my_ensemble.ensemble_comm.rank == 1:

        rhs = Function(fs).vector()
        rhs.set_local(0.5*np.ones(fc.get_nx_local()))

        result = _solve_forcing_covariance(fc, A, rhs)

        result_actual = result.gather()

        result_expected = np.linalg.solve(A_numpy, 0.5*np.ones(nx + 1))
        result_expected = np.dot(cov, result_expected)
        result_expected = np.linalg.solve(A_numpy, result_expected)

        assert_allclose(result_expected, result_actual, atol = 1.e-10)
