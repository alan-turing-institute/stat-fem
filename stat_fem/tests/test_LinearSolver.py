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

import gc
gc.disable()

@pytest.fixture
def ls(A, b, fc, od):
    return LinearSolver(A, b, fc, od)

@pytest.fixture
def ls_parallel(A, b, fc, od, my_ensemble):
    return LinearSolver(A, b, fc, od, ensemble_comm=my_ensemble.ensemble_comm)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_LinearSolver_solve_posterior(fs, A, b, meshcoords, fc, od, interp, Ks, params, ls):
    "test solve_conditioned_FEM"

    rho = np.exp(params[0])

    # solve and put solution in u

    u = Function(fs)
    ls.set_params(params)
    ls.solve_posterior(u)
    u_f = u.vector().gather()

    u_scaled = Function(fs)
    ls.solve_posterior(u_scaled, scale_mean=True)
    u_f_scaled = u_scaled.vector().gather()

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
        assert_allclose(u_expected*rho, u_f_scaled, atol=1.e-10)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_solve_posterior_parallel(my_ensemble, fs, A, b, meshcoords, fc, od, interp, Ks, params, ls_parallel):
    "test solve_conditioned_FEM"

    # solve and put solution in u

    u = Function(fs)
    ls_parallel.set_params(params)
    ls_parallel.solve_posterior(u)
    u_f = u.vector().gather()

    u_scaled = Function(fs)
    ls_parallel.solve_posterior(u_scaled, scale_mean=True)
    u_f_scaled = u_scaled.vector().gather()
    
    rho = np.exp(params[0])

    mu, Cu = ls_parallel.solve_prior()

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
        assert_allclose(u_expected*rho, u_f_scaled, atol=1.e-10)
    elif my_ensemble.ensemble_comm.rank != 0:
        assert_allclose(u_f, np.zeros(u_f.shape))

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_solve_posterior_covariance(A, b, fc, od, params, Ks, ls):
    "test solve_posterior_covariance"

    ls.set_params(params)
    muy, Cuy = ls.solve_posterior_covariance()
    muy_scaled, _ = ls.solve_posterior_covariance(scale_mean=True)
    
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
        assert_allclose(muy_scaled, muy_expected*rho, atol=1.e-10)
    else:
        assert muy.shape == (0,)
        assert Cuy.shape == (0, 0)
        assert muy2.shape == (0,)
        assert Cuy2.shape == (0, 0)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_LinearSolver_solve_posterior_covariance_parallel(my_ensemble, A, b, fc, od, params, Ks, ls_parallel):
    "test solve_posterior_covariance"

    ls_parallel.set_params(params)
    muy, Cuy = ls_parallel.solve_posterior_covariance()
    muy_scaled, _ = ls_parallel.solve_posterior_covariance(scale_mean=True)
    
    rho = np.exp(params[0])

    mu, Cu = ls_parallel.solve_prior()

    if COMM_WORLD.rank == 0:
        Kinv = np.linalg.inv(Ks)
        Cuy_expected = np.linalg.inv(np.linalg.inv(Cu) + rho**2*Kinv)
        muy_expected = np.dot(Cuy_expected, rho*np.dot(Kinv, od.get_data()) +
                                            np.linalg.solve(Cu, mu))
        assert_allclose(muy, muy_expected, atol=1.e-10)
        assert_allclose(Cuy, Cuy_expected, atol=1.e-10)
        assert_allclose(muy_scaled, muy_expected*rho, atol=1.e-10)
    else:
        assert muy.shape == (0,)
        assert Cuy.shape == (0, 0)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_LinearSolver_solve_prior(fs, A, b, fc, od, interp, cov, A_numpy, ls):
    "test solve_conditioned_FEM"

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
def test_LinearSolver_solve_prior_parallel(my_ensemble, fs, A, b, fc, od, interp, cov, A_numpy, ls_parallel):
    "test solve_conditioned_FEM"

    mu, Cu = ls_parallel.solve_prior()

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
def test_LinearSolver_solve_prior_generating(fs, A, b, fc, od, params, interp, A_numpy, cov, K, ls):
    "test the function to solve the prior of the generating process"

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
def test_solve_posterior_generating(fs, A, b, fc, od, params, ls):
    "test the function to solve the posterior of the generating process"

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
def test_predict_mean(fs, A, b, fc, od, interp, params, coords_predict, interp_predict, ls):
    "test the function to predict the mean at new sensor locations"

    ls.set_params(params)
    mu_actual = ls.predict_mean(coords_predict)
    mu_actual_unscaled = ls.predict_mean(coords_predict, scale_mean=False)
    
    mu_actual2 = predict_mean(A, b, fc, od, params, coords_predict)

    rho = np.exp(params[0])

    u = Function(fs)
    ls.solve_posterior(u)
    u_f = u.vector().gather()

    if COMM_WORLD.rank == 0:
        mu_expect = rho*np.dot(interp_predict.T, u_f)
        assert_allclose(mu_actual, mu_expect)
        assert_allclose(mu_actual2, mu_expect)
        assert_allclose(mu_actual_unscaled, mu_expect/rho)
    else:
        assert mu_actual.shape == (0,)
        assert mu_actual2.shape == (0,)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_predict_covariance(fs, A, b, meshcoords, fc, od, interp, Ks, params, coords_predict, interp_predict, Ks_predict, ls):
    "test the function to predict the mean at new sensor locations"

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

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_LinearSolver_logposterior(A, b, fc, od, params, Ks, ls):
    "test the loglikelihood method"

    rho = np.exp(params[0])
    
    mu, Cu = ls.solve_prior()

    loglike_actual = ls.logposterior(params)

    if COMM_WORLD.rank == 0:
        KCu = rho**2*Cu + Ks
        loglike_expected = 0.5*(np.linalg.multi_dot([od.get_data() - rho*mu,
                                                     np.linalg.inv(KCu),
                                                     od.get_data() - rho*mu]) +
                                np.log(np.linalg.det(KCu)) +
                                od.get_n_obs()*np.log(2.*np.pi))
    else:
        loglike_expected = None

    loglike_expected = COMM_WORLD.bcast(loglike_expected, root=0)

    assert_allclose(loglike_expected, loglike_actual)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_LinearSolver_logpost_deriv(A, b, fc, od, params, Ks, ls):
    "test the model loglikelihood using finite differences"

    dx = 1.e-8

    loglike_deriv_actual = ls.logpost_deriv(params)

    loglike_deriv_fd = np.zeros(3)

    loglike_deriv_fd[0] = (ls.logposterior(params + np.array([dx, 0., 0.])) -
                           ls.logposterior(params                         ))/dx
    loglike_deriv_fd[1] = (ls.logposterior(params + np.array([0., dx, 0.])) -
                           ls.logposterior(params                         ))/dx
    loglike_deriv_fd[2] = (ls.logposterior(params + np.array([0., 0., dx])) -
                           ls.logposterior(params                         ))/dx

    assert_allclose(loglike_deriv_actual, loglike_deriv_fd, atol=1.e-6, rtol=1.e-6)

gc.collect()
