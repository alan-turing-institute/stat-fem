import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from firedrake.ensemble import Ensemble
from firedrake.function import Function
from firedrake import COMM_WORLD
import pytest
from ..ForcingCovariance import ForcingCovariance
from ..solving_utils import _solve_forcing_covariance
from .helper_funcs import create_assembled_problem, create_forcing_covariance, create_problem_numpy

def test_solve_forcing_covariance():
    "test solve_forcing_covariance"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    # fc, cov = create_forcing_covariance(mesh, V)
    #
    # ab, _ = create_problem_numpy(mesh, V)
    #
    # rhs = Function(V).vector()
    # rhs.set_local(np.ones(fc.get_nx_local()))

#     result = _solve_forcing_covariance(fc, A, rhs)
#
#     result_actual = result.gather()
#
#     result_expected = np.linalg.solve(ab, np.ones(nx + 1))
#     result_expected = np.dot(cov, result_expected)
#     result_expected = np.linalg.solve(ab, result_expected)
#
#     assert_allclose(result_expected, result_actual, atol = 1.e-10)

# @pytest.mark.mpi
# @pytest.mark.parametrize("n_proc", [1, 2])
# def test_solve_forcing_covariance_parallel(n_proc):
#     "test that solve_forcing_covariance can be called independently from an ensemble process"
#
#     nx = 10
#
#     my_ensemble = Ensemble(COMM_WORLD, n_proc)
#
#     A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)
#
#     fc, cov = create_forcing_covariance(mesh, V)
#
#     ab, _ = create_problem_numpy(mesh, V)
#
#     if my_ensemble.ensemble_comm.rank == 0:
#
#         rhs = Function(V).vector()
#         rhs.set_local(np.ones(fc.get_nx_local()))
#
#         result = _solve_forcing_covariance(fc, A, rhs)
#
#         result_actual = result.gather()
#
#         result_expected = np.linalg.solve(ab, np.ones(nx + 1))
#         result_expected = np.dot(cov, result_expected)
#         result_expected = np.linalg.solve(ab, result_expected)
#
#         assert_allclose(result_expected, result_actual, atol = 1.e-10)
#
#     elif my_ensemble.ensemble_comm.rank == 1:
#
#         rhs = Function(V).vector()
#         rhs.set_local(0.5*np.ones(fc.get_nx_local()))
#
#         result = _solve_forcing_covariance(fc, A, rhs)
#
#         result_actual = result.gather()
#
#         result_expected = np.linalg.solve(ab, 0.5*np.ones(nx + 1))
#         result_expected = np.dot(cov, result_expected)
#         result_expected = np.linalg.solve(ab, result_expected)
#
#         assert_allclose(result_expected, result_actual, atol = 1.e-10)
