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
from .test_shared import create_K_plus_sigma, create_meshcoords

def test_solve_posterior():
    "test solve_conditioned_FEM"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    # solve and put solution in u

    u = Function(V)
    solve_posterior(A, u, b, fc, od, np.ones(3))
    u_f = u.vector().gather()

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3))

    # need "data" on actual FEM grid to get full Cu

    meshcoords = create_meshcoords(mesh, V)

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    mu_full, Cu_full = solve_prior_covariance(A, b, fc, full_data, np.ones(3))

    Ks = create_K_plus_sigma(1., 1.)

    if COMM_WORLD.rank == 0:
        tmp_1 = np.linalg.solve(Ks, od.get_data())
        tmp_1 = np.linalg.multi_dot([Cu_full, interp, tmp_1]) + mu_full
        KCinv = np.linalg.inv(Ks + Cu)
        tmp_2 = np.linalg.multi_dot([Cu_full, interp, KCinv, interp.T, tmp_1])
        u_expected = tmp_1 - tmp_2
        assert_allclose(u_expected, u_f, atol=1.e-10)

    fc.destroy()

def helper_solve_posterior_parallel(n_proc):
    "test solve_conditioned_FEM"

    nx = 10

    my_ensemble = Ensemble(COMM_WORLD, n_proc)

    A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    # solve and put solution in u

    u = Function(V)
    solve_posterior(A, u, b, fc, od, np.ones(3), my_ensemble.ensemble_comm)
    u_f = u.vector().gather()

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3))

    # need "data" on actual FEM grid to get full Cu

    meshcoords = create_meshcoords(mesh, V)

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    mu_full, Cu_full = solve_prior_covariance(A, b, fc, full_data, np.ones(3))

    Ks = create_K_plus_sigma(1., 1.)

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        tmp_1 = np.linalg.solve(Ks, od.get_data())
        tmp_1 = np.linalg.multi_dot([Cu_full, interp, tmp_1]) + mu_full
        KCinv = np.linalg.inv(Ks + Cu)
        tmp_2 = np.linalg.multi_dot([Cu_full, interp, KCinv, interp.T, tmp_1])
        u_expected = tmp_1 - tmp_2
        assert_allclose(u_expected, u_f, atol=1.e-10)
    elif my_ensemble.ensemble_comm.rank != 0:
        assert_allclose(u_f, np.zeros(u_f.shape))

    fc.destroy()

@pytest.mark.mpi
def test_solve_posterior_parallel_1():
    helper_solve_posterior_parallel(1)

@pytest.mark.mpi
def test_solve_posterior_parallel_2():
    helper_solve_posterior_parallel(2)

def test_solve_posterior_covariance():
    "test solve_posterior_covariance"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3))
    muy, Cuy = solve_posterior_covariance(A, b, fc, od, np.ones(3))

    Kinv = np.linalg.inv(create_K_plus_sigma(1., 1.))

    if COMM_WORLD.rank == 0:
        Cuy_expected = np.linalg.inv(np.linalg.inv(Cu) + Kinv)
        muy_expected = np.dot(Cuy_expected, np.dot(Kinv, od.get_data()) +
                                            np.linalg.solve(Cu, mu))
        assert_allclose(muy, muy_expected, atol=1.e-10)
        assert_allclose(Cuy, Cuy_expected, atol=1.e-10)
    else:
        assert muy.shape == (0,)
        assert Cuy.shape == (0, 0)

    fc.destroy()

@pytest.mark.mpi
def test_solve_posterior_covariance_parallel_1():
    helper_solve_posterior_covariance_parallel(1)

@pytest.mark.mpi
def test_solve_posterior_covariance_parallel_2():
    helper_solve_posterior_covariance_parallel(2)

def helper_solve_posterior_covariance_parallel(n_proc):
    "test solve_posterior_covariance"

    nx = 10

    my_ensemble = Ensemble(COMM_WORLD, n_proc)

    A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3))
    muy, Cuy = solve_posterior_covariance(A, b, fc, od, np.ones(3), my_ensemble.ensemble_comm)

    Kinv = np.linalg.inv(create_K_plus_sigma(1., 1.))

    if COMM_WORLD.rank == 0:
        Cuy_expected = np.linalg.inv(np.linalg.inv(Cu) + Kinv)
        muy_expected = np.dot(Cuy_expected, np.dot(Kinv, od.get_data()) +
                                            np.linalg.solve(Cu, mu))
        assert_allclose(muy, muy_expected, atol=1.e-10)
        assert_allclose(Cuy, Cuy_expected, atol=1.e-10)
    else:
        assert muy.shape == (0,)
        assert Cuy.shape == (0, 0)

    fc.destroy()

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

    fc.destroy()

@pytest.mark.mpi
def test_solve_prior_covariance_parallel_1():
    helper_solve_prior_covariance_parallel(1)

@pytest.mark.mpi
def test_solve_prior_covariance_parallel_2():
    helper_solve_prior_covariance_parallel(2)

def helper_solve_prior_covariance_parallel(n_proc):
    "test solve_conditioned_FEM"

    nx = 10

    my_ensemble = Ensemble(COMM_WORLD, n_proc)

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

    fc.destroy()