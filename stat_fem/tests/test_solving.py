import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from firedrake import UnitIntervalMesh, FunctionSpace, dx, TrialFunction, TestFunction, interpolate
from firedrake import dot, assemble, DirichletBC, grad, Function, VectorFunctionSpace
from firedrake import COMM_WORLD, Ensemble, SpatialCoordinate, pi, sin, solve
from ..solving import solve_posterior, solve_posterior_covariance, solve_prior_covariance
from ..ForcingCovariance import ForcingCovariance
from ..ObsData import ObsData
from .test_shared import create_assembled_problem, create_interp
from .test_shared import create_obs_data, create_problem_numpy, create_forcing_covariance

def test_solve_posterior():
    "test solve_conditioned_FEM"

    pass

def test_solve_posterior_covariance():
    "test solve_conditioned_FEM"

    pass

def test_solve_prior_covariance():
    "test solve_conditioned_FEM"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3))

    C_expected = np.linalg.solve(ab, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(ab, C_expected)
    C_expected = np.dot(interp.T, C_expected)

    u = Function(V)
    solve(A, u, b)
    m_expected = np.dot(interp.T, u.vector().gather())

    if COMM_WORLD.rank == 0:
        assert_allclose(m_expected, mu, atol = 1.e-10)
        assert_allclose(C_expected, Cu, atol = 1.e-10)
    else:
        assert mu.shape == (0,)
        assert Cu.shape == (0,0)

@pytest.mark.mpi
def test_solve_prior_covariance_parallel():
    "test solve_conditioned_FEM"

    nx = 10

    my_ensemble = Ensemble(COMM_WORLD, 2)

    A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3), my_ensemble.ensemble_comm)

    C_expected = np.linalg.solve(ab, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(ab, C_expected)
    C_expected = np.dot(interp.T, C_expected)

    u = Function(V)
    solve(A, u, b)
    m_expected = np.dot(interp.T, u.vector().gather())

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        assert_allclose(m_expected, mu, atol = 1.e-10)
        assert_allclose(C_expected, Cu, atol = 1.e-10)
    else:
        assert mu.shape == (0,)
        assert Cu.shape == (0,0)