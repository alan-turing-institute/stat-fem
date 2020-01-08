import pytest
import numpy as np
from numpy.testing import assert_allclose
from firedrake import Function, COMM_WORLD, COMM_SELF, Ensemble, solve
from ..solving import solve_posterior, solve_posterior_covariance, solve_prior_covariance
from ..solving import solve_prior_generating, solve_posterior_generating
from ..ForcingCovariance import ForcingCovariance
from ..ObsData import ObsData
from .helper_funcs import create_assembled_problem, create_interp
from .helper_funcs import create_obs_data, create_problem_numpy, create_forcing_covariance
from .helper_funcs import create_K_plus_sigma, create_meshcoords, create_K

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
    solve_posterior(A, u, b, fc, od, np.zeros(3))
    u_f = u.vector().gather()

    mu, Cu = solve_prior_covariance(A, b, fc, od)

    # need "data" on actual FEM grid to get full Cu

    meshcoords = create_meshcoords(mesh, V)

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    mu_full, Cu_full = solve_prior_covariance(A, b, fc, full_data)

    Ks = create_K_plus_sigma(np.log(1.), np.log(1.))

    if COMM_WORLD.rank == 0:
        tmp_1 = np.linalg.solve(Ks, od.get_data())
        tmp_1 = np.linalg.multi_dot([Cu_full, interp, tmp_1]) + mu_full
        KCinv = np.linalg.inv(Ks + Cu)
        tmp_2 = np.linalg.multi_dot([Cu_full, interp, KCinv, interp.T, tmp_1])
        u_expected = tmp_1 - tmp_2
        assert_allclose(u_expected, u_f, atol=1.e-10)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("n_proc", [1, 2])
def test_solve_posterior_parallel(n_proc):
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
    solve_posterior(A, u, b, fc, od, np.zeros(3), my_ensemble.ensemble_comm)
    u_f = u.vector().gather()

    mu, Cu = solve_prior_covariance(A, b, fc, od, my_ensemble.ensemble_comm)

    # need "data" on actual FEM grid to get full Cu

    meshcoords = create_meshcoords(mesh, V)

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    mu_full, Cu_full = solve_prior_covariance(A, b, fc, full_data, my_ensemble.ensemble_comm)

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        Ks = create_K_plus_sigma(np.log(1.), np.log(1.))
        tmp_1 = np.linalg.solve(Ks, od.get_data())
        tmp_1 = np.linalg.multi_dot([Cu_full, interp, tmp_1]) + mu_full
        KCinv = np.linalg.inv(Ks + Cu)
        tmp_2 = np.linalg.multi_dot([Cu_full, interp, KCinv, interp.T, tmp_1])
        u_expected = tmp_1 - tmp_2
        assert_allclose(u_expected, u_f, atol=1.e-10)
    elif my_ensemble.ensemble_comm.rank != 0:
        assert_allclose(u_f, np.zeros(u_f.shape))

    fc.destroy()

def test_solve_posterior_covariance():
    "test solve_posterior_covariance"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od)
    muy, Cuy = solve_posterior_covariance(A, b, fc, od, np.zeros(3))

    Kinv = np.linalg.inv(create_K_plus_sigma(np.log(1.), np.log(1.)))

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
@pytest.mark.parametrize("n_proc",[1, 2])
def test_solve_posterior_covariance_parallel(n_proc):
    "test solve_posterior_covariance"

    nx = 10

    my_ensemble = Ensemble(COMM_WORLD, n_proc)

    A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab,_ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od, my_ensemble.ensemble_comm)
    muy, Cuy = solve_posterior_covariance(A, b, fc, od, np.zeros(3), my_ensemble.ensemble_comm)

    if COMM_WORLD.rank == 0:
        Kinv = np.linalg.inv(create_K_plus_sigma(np.log(1.), np.log(1.)))
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

    mu, Cu = solve_prior_covariance(A, b, fc, od)

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
@pytest.mark.parametrize("n_proc",[1, 2])
def test_solve_prior_covariance_parallel(n_proc):
    "test solve_conditioned_FEM"

    nx = 10

    my_ensemble = Ensemble(COMM_WORLD, n_proc)

    A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od, my_ensemble.ensemble_comm)

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

    if COMM_WORLD.size == 2:
        assert False

def test_solve_prior_generating():
    "test the function to solve the prior of the generating process"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    mu, Cu = solve_prior_covariance(A, b, fc, od)
    m_eta, C_eta = solve_prior_generating(A, b, fc, od, np.zeros(3))

    rho = 1.

    C_expected = np.linalg.solve(ab, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(ab, C_expected)
    C_expected = rho**2*np.dot(interp.T, C_expected) + create_K(0., 0.)

    u = Function(V)
    solve(A, u, b)
    m_expected = rho*np.dot(interp.T, u.vector().gather())

    if COMM_WORLD.rank == 0:
        assert_allclose(m_expected, m_eta, atol = 1.e-10)
        assert_allclose(C_expected, C_eta, atol = 1.e-10)
    else:
        assert m_eta.shape == (0,)
        assert C_eta.shape == (0,0)

    fc.destroy()

def test_solve_posterior_generating():
    "test the function to solve the posterior of the generating process"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    ab, _ = create_problem_numpy(mesh, V)

    interp = create_interp(mesh, V)

    m_eta, C_eta = solve_prior_generating(A, b, fc, od, np.zeros(3))
    m_etay, C_etay = solve_posterior_generating(A, b, fc, od, np.zeros(3))

    rho = 1.

    if COMM_WORLD.rank == 0:
        C_expected = np.linalg.inv(C_eta)
        C_expected = C_expected + np.eye(od.get_n_obs())/0.1**2
        C_expected = np.linalg.inv(C_expected)
        m_expected = np.dot(C_expected, od.get_data()/0.1**2 + np.linalg.solve(C_eta, m_eta))
        assert_allclose(m_expected, m_etay, atol = 1.e-10)
        assert_allclose(C_expected, C_etay, atol = 1.e-10)
    else:
        assert m_etay.shape == (0,)
        assert C_etay.shape == (0,0)

    fc.destroy()