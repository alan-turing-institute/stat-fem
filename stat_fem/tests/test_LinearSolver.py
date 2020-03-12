import pytest
import numpy as np
from numpy.testing import assert_allclose
from firedrake import Function, COMM_WORLD, solve
from ..LinearSolver import LinearSolver
from ..ObsData import ObsData
from ..solving import solve_prior_covariance, solve_posterior, solve_posterior_covariance
from ..solving import solve_prior_generating, solve_posterior_generating
from ..solving import predict_mean, predict_covariance
from .helper_funcs import nx, params, my_ensemble, comm, mesh, fs, A, b, meshcoords, fc, od, interp, Ks
from .helper_funcs import A_numpy, cov, K, coords, coords_predict, interp_predict, Ks_predict

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_LinearSolver_solve_posterior(fs, A, b, meshcoords, fc, od, interp, Ks, params):
    "test solve_conditioned_FEM"

    rho = np.exp(params[0])

    # solve and put solution in u

    u = Function(fs)
    ls = LinearSolver(A, b, fc, od)
    ls.set_params(params)
    ls.solve_posterior(u)
    u_f = u.vector().gather()

    mu, Cu = ls.solve_prior()

    u2 = Function(fs)

    solve_posterior(A, u2, b, fc, od, params)
    u_f2 = u2.vector().gather()

    # need "data" on actual FEM grid to get full Cu

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    mu_full, Cu_full = solve_prior_covariance(A, b, fc, full_data)

    if COMM_WORLD.rank == 0:
        tmp_1 = np.linalg.solve(Ks, od.get_data())
        tmp_1 = rho*np.linalg.multi_dot([Cu_full, interp, tmp_1]) + mu_full
        KCinv = np.linalg.inv(Ks/rho**2 + Cu)
        tmp_2 = np.linalg.multi_dot([Cu_full, interp, KCinv, interp.T, tmp_1])
        u_expected = tmp_1 - tmp_2
        assert_allclose(u_expected, u_f, atol=1.e-10)
        assert_allclose(u_expected, u_f2, atol=1.e-10)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_solve_posterior_parallel(my_ensemble, fs, A, b, meshcoords, fc, od, interp, Ks, params):
    "test solve_conditioned_FEM"

    # solve and put solution in u

    u = Function(fs)
    ls = LinearSolver(A, b, fc, od, ensemble_comm=my_ensemble.ensemble_comm)
    ls.set_params(params)
    ls.solve_posterior(u)
    u_f = u.vector().gather()

    rho = np.exp(params[0])

    mu, Cu = ls.solve_prior()

    # need "data" on actual FEM grid to get full Cu

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    ls2 = LinearSolver(A, b, fc, full_data, ensemble_comm=my_ensemble.ensemble_comm)
    mu_full, Cu_full = ls2.solve_prior()

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        tmp_1 = np.linalg.solve(Ks, od.get_data())
        tmp_1 = rho*np.linalg.multi_dot([Cu_full, interp, tmp_1]) + mu_full
        KCinv = np.linalg.inv(Ks + rho**2*Cu)
        tmp_2 = np.linalg.multi_dot([Cu_full, interp, KCinv, interp.T, tmp_1])
        u_expected = tmp_1 - rho**2*tmp_2
        assert_allclose(u_expected, u_f, atol=1.e-10)
    elif my_ensemble.ensemble_comm.rank != 0:
        assert_allclose(u_f, np.zeros(u_f.shape))

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_solve_posterior_covariance(A, b, fc, od, params, Ks):
    "test solve_posterior_covariance"

    ls = LinearSolver(A, b, fc, od)
    ls.set_params(params)
    muy, Cuy = ls.solve_posterior_covariance()

    muy2, Cuy2 = solve_posterior_covariance(A, b, fc, od, params)

    rho = np.exp(params[0])

    mu, Cu = ls.solve_prior()

    Kinv = np.linalg.inv(Ks)

    if COMM_WORLD.rank == 0:
        Cuy_expected = np.linalg.inv(np.linalg.inv(Cu) + rho**2*Kinv)
        muy_expected = np.dot(Cuy_expected, rho*np.dot(Kinv, od.get_data()) +
                                            np.linalg.solve(Cu, mu))
        assert_allclose(muy, muy_expected, atol=1.e-10)
        assert_allclose(Cuy, Cuy_expected, atol=1.e-10)
        assert_allclose(muy2, muy_expected, atol=1.e-10)
        assert_allclose(Cuy2, Cuy_expected, atol=1.e-10)
    else:
        assert muy.shape == (0,)
        assert Cuy.shape == (0, 0)
        assert muy2.shape == (0,)
        assert Cuy2.shape == (0, 0)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_LinearSolver_solve_posterior_covariance_parallel(my_ensemble, A, b, fc, od, params, Ks):
    "test solve_posterior_covariance"

    ls = LinearSolver(A, b, fc, od, ensemble_comm=my_ensemble.ensemble_comm)
    ls.set_params(params)
    muy, Cuy = ls.solve_posterior_covariance()

    rho = np.exp(params[0])

    mu, Cu = ls.solve_prior()

    if COMM_WORLD.rank == 0:
        Kinv = np.linalg.inv(Ks)
        Cuy_expected = np.linalg.inv(np.linalg.inv(Cu) + rho**2*Kinv)
        muy_expected = np.dot(Cuy_expected, rho*np.dot(Kinv, od.get_data()) +
                                            np.linalg.solve(Cu, mu))
        assert_allclose(muy, muy_expected, atol=1.e-10)
        assert_allclose(Cuy, Cuy_expected, atol=1.e-10)
    else:
        assert muy.shape == (0,)
        assert Cuy.shape == (0, 0)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_LinearSolver_solve_prior(fs, A, b, fc, od, interp, cov, A_numpy):
    "test solve_conditioned_FEM"

    ls = LinearSolver(A, b, fc, od)
    mu, Cu = ls.solve_prior()

    mu2, Cu2 = solve_prior_covariance(A, b, fc, od)

    C_expected = np.linalg.solve(A_numpy, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(A_numpy, C_expected)
    C_expected = np.dot(interp.T, C_expected)

    u = Function(fs)
    solve(A, u, b)
    m_expected = np.dot(interp.T, u.vector().gather())

    if COMM_WORLD.rank == 0:
        assert_allclose(m_expected, mu, atol = 1.e-10)
        assert_allclose(C_expected, Cu, atol = 1.e-10)
        assert_allclose(m_expected, mu2, atol = 1.e-10)
        assert_allclose(C_expected, Cu2, atol = 1.e-10)
    else:
        assert mu.shape == (0,)
        assert Cu.shape == (0,0)
        assert mu2.shape == (0,)
        assert Cu2.shape == (0,0)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_LinearSolver_solve_prior_parallel(my_ensemble, fs, A, b, fc, od, interp, cov, A_numpy):
    "test solve_conditioned_FEM"

    ls = LinearSolver(A, b, fc, od, ensemble_comm=my_ensemble.ensemble_comm)
    mu, Cu = ls.solve_prior()

    C_expected = np.linalg.solve(A_numpy, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(A_numpy, C_expected)
    C_expected = np.dot(interp.T, C_expected)

    u = Function(fs)
    solve(A, u, b)
    m_expected = np.dot(interp.T, u.vector().gather())

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        assert_allclose(m_expected, mu, atol = 1.e-10)
        assert_allclose(C_expected, Cu, atol = 1.e-10)
    else:
        assert mu.shape == (0,)
        assert Cu.shape == (0,0)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_LinearSolver_solve_prior_generating(fs, A, b, fc, od, params, interp, A_numpy, cov, K):
    "test the function to solve the prior of the generating process"

    ls = LinearSolver(A, b, fc, od)
    ls.set_params(params)
    m_eta, C_eta = ls.solve_prior_generating()

    m_eta2, C_eta2 = solve_prior_generating(A, b, fc, od, params)

    rho = np.exp(params[0])

    C_expected = np.linalg.solve(A_numpy, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(A_numpy, C_expected)
    C_expected = rho**2*np.dot(interp.T, C_expected) + K

    u = Function(fs)
    solve(A, u, b)
    m_expected = rho*np.dot(interp.T, u.vector().gather())

    if COMM_WORLD.rank == 0:
        assert_allclose(m_expected, m_eta, atol = 1.e-10)
        assert_allclose(C_expected, C_eta, atol = 1.e-10)
        assert_allclose(m_expected, m_eta2, atol = 1.e-10)
        assert_allclose(C_expected, C_eta2, atol = 1.e-10)
    else:
        assert m_eta.shape == (0,)
        assert C_eta.shape == (0,0)
        assert m_eta2.shape == (0,)
        assert C_eta2.shape == (0,0)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_solve_posterior_generating(fs, A, b, fc, od, params):
    "test the function to solve the posterior of the generating process"

    ls = LinearSolver(A, b, fc, od)
    ls.set_params(params)
    m_eta, C_eta = ls.solve_prior_generating()
    m_etay, C_etay = ls.solve_posterior_generating()

    m_etay2, C_etay2 = solve_posterior_generating(A, b, fc, od, params)

    rho = np.exp(params[0])

    if COMM_WORLD.rank == 0:
        C_expected = np.linalg.inv(C_eta)
        C_expected = C_expected + np.eye(od.get_n_obs())/od.get_unc()**2
        C_expected = np.linalg.inv(C_expected)
        m_expected = np.dot(C_expected, od.get_data()/od.get_unc()**2 + np.linalg.solve(C_eta, m_eta))
        assert_allclose(m_expected, m_etay, atol = 1.e-10)
        assert_allclose(C_expected, C_etay, atol = 1.e-10)
        assert_allclose(m_expected, m_etay2, atol = 1.e-10)
        assert_allclose(C_expected, C_etay2, atol = 1.e-10)
    else:
        assert m_etay.shape == (0,)
        assert C_etay.shape == (0,0)
        assert m_etay2.shape == (0,)
        assert C_etay2.shape == (0,0)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_predict_mean(fs, A, b, fc, od, interp, params, coords_predict, interp_predict):
    "test the function to predict the mean at new sensor locations"

    ls = LinearSolver(A, b, fc, od)
    ls.set_params(params)
    mu_actual = ls.predict_mean(coords_predict)

    mu_actual2 = predict_mean(A, b, fc, od, params, coords_predict)

    rho = np.exp(params[0])

    u = Function(fs)
    ls.solve_posterior(u)
    u_f = u.vector().gather()

    if COMM_WORLD.rank == 0:
        mu_expect = rho*np.dot(interp_predict.T, u_f)
        assert_allclose(mu_actual, mu_expect)
        assert_allclose(mu_actual2, mu_expect)
    else:
        assert mu_actual.shape == (0,)
        assert mu_actual2.shape == (0,)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_predict_covariance(fs, A, b, meshcoords, fc, od, interp, Ks, params, coords_predict, interp_predict, Ks_predict):
    "test the function to predict the mean at new sensor locations"

    ls = LinearSolver(A, b, fc, od)
    ls.set_params(params)
    Cu_actual = ls.predict_covariance(coords_predict, 0.1)
    Cu_actual2 = predict_covariance(A, b, fc, od, params, coords_predict, 0.1)

    rho = np.exp(params[0])

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    ls2 = LinearSolver(A, b, fc, full_data)
    mu_full, Cu_full = ls2.solve_prior()

    if COMM_WORLD.rank == 0:
        Cuinv = np.linalg.inv(Cu_full)
        Kinv = np.linalg.multi_dot([interp, np.linalg.inv(Ks), interp.T])
        Cu_expect = (Ks_predict +
                     rho**2*np.linalg.multi_dot([interp_predict.T, np.linalg.inv(Cuinv + rho**2*Kinv), interp_predict]))
        assert_allclose(Cu_actual, Cu_expect, atol=1.e-3, rtol=1.e-3)
        assert_allclose(Cu_actual2, Cu_expect, atol=1.e-3, rtol=1.e-3)
    else:
        assert Cu_actual.shape == (0,0)
        assert Cu_actual2.shape == (0,0)

    Cu_actual = ls.predict_covariance(coords_predict.flatten(), 0.1)

    if COMM_WORLD.rank == 0:
        assert_allclose(Cu_actual, Cu_expect, atol=1.e-3, rtol=1.e-3)

    with pytest.raises(AssertionError):
        ls.predict_covariance(np.array([[0.2, 0.2]]), 0.1)
