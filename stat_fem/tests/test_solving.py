import pytest
import numpy as np
from numpy.testing import assert_allclose
from firedrake import Function, COMM_WORLD, solve
from ..solving import solve_posterior, solve_posterior_covariance, solve_prior_covariance
from ..solving import solve_prior_generating, solve_posterior_generating, predict_mean, predict_covariance
from ..ObsData import ObsData
from .helper_funcs import nx, params, my_ensemble, comm, mesh, fs, A, b, meshcoords, fc, od, interp, Ks
from .helper_funcs import A_numpy, cov, K, coords, coords_predict, interp_predict, Ks_predict

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_solve_posterior(fs, A, b, meshcoords, fc, od, interp, Ks, params):
    "test solve_conditioned_FEM"

    rho = np.exp(params[0])

    # solve and put solution in u

    u = Function(fs)
    solve_posterior(A, u, b, fc, od, params)
    u_f = u.vector().gather()

    mu, Cu = solve_prior_covariance(A, b, fc, od)

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

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_solve_posterior_parallel(my_ensemble, fs, A, b, meshcoords, fc, od, interp, Ks):
    "test solve_conditioned_FEM"

    # solve and put solution in u

    u = Function(fs)
    solve_posterior(A, u, b, fc, od, np.zeros(3), my_ensemble.ensemble_comm)
    u_f = u.vector().gather()

    mu, Cu = solve_prior_covariance(A, b, fc, od, my_ensemble.ensemble_comm)

    # need "data" on actual FEM grid to get full Cu

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    mu_full, Cu_full = solve_prior_covariance(A, b, fc, full_data, my_ensemble.ensemble_comm)

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

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_solve_posterior_covariance(A, b, fc, od, params, Ks):
    "test solve_posterior_covariance"

    rho = np.exp(params[0])

    mu, Cu = solve_prior_covariance(A, b, fc, od)
    muy, Cuy = solve_posterior_covariance(A, b, fc, od, params)

    Kinv = np.linalg.inv(Ks)

    if COMM_WORLD.rank == 0:
        Cuy_expected = np.linalg.inv(np.linalg.inv(Cu) + rho**2*Kinv)
        muy_expected = np.dot(Cuy_expected, rho*np.dot(Kinv, od.get_data()) +
                                            np.linalg.solve(Cu, mu))
        assert_allclose(muy, muy_expected, atol=1.e-10)
        assert_allclose(Cuy, Cuy_expected, atol=1.e-10)
    else:
        assert muy.shape == (0,)
        assert Cuy.shape == (0, 0)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_solve_posterior_covariance_parallel(my_ensemble, A, b, fc, od, params, Ks):
    "test solve_posterior_covariance"

    rho = np.exp(params[0])

    mu, Cu = solve_prior_covariance(A, b, fc, od, my_ensemble.ensemble_comm)
    muy, Cuy = solve_posterior_covariance(A, b, fc, od, params, my_ensemble.ensemble_comm)

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
def test_solve_prior_covariance(fs, A, b, fc, od, interp, cov, A_numpy):
    "test solve_conditioned_FEM"

    mu, Cu = solve_prior_covariance(A, b, fc, od)

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
    else:
        assert mu.shape == (0,)
        assert Cu.shape == (0,0)

    fc.destroy()

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
@pytest.mark.parametrize("coords", [1], indirect=["coords"])
def test_solve_prior_covariance_parallel(my_ensemble, fs, A, b, fc, od, interp, cov, A_numpy):
    "test solve_conditioned_FEM"

    mu, Cu = solve_prior_covariance(A, b, fc, od, my_ensemble.ensemble_comm)

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
def test_solve_prior_generating(fs, A, b, fc, od, params, interp, A_numpy, cov, K):
    "test the function to solve the prior of the generating process"

    mu, Cu = solve_prior_covariance(A, b, fc, od)
    m_eta, C_eta = solve_prior_generating(A, b, fc, od, params)

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
    else:
        assert m_eta.shape == (0,)
        assert C_eta.shape == (0,0)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_solve_posterior_generating(fs, A, b, fc, od, params):
    "test the function to solve the posterior of the generating process"

    m_eta, C_eta = solve_prior_generating(A, b, fc, od, params)
    m_etay, C_etay = solve_posterior_generating(A, b, fc, od, params)

    rho = np.exp(params[0])

    if COMM_WORLD.rank == 0:
        C_expected = np.linalg.inv(C_eta)
        C_expected = C_expected + np.eye(od.get_n_obs())/od.get_unc()**2
        C_expected = np.linalg.inv(C_expected)
        m_expected = np.dot(C_expected, od.get_data()/od.get_unc()**2 + np.linalg.solve(C_eta, m_eta))
        assert_allclose(m_expected, m_etay, atol = 1.e-10)
        assert_allclose(C_expected, C_etay, atol = 1.e-10)
    else:
        assert m_etay.shape == (0,)
        assert C_etay.shape == (0,0)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_predict_mean(fs, A, b, fc, od, interp, params, coords_predict, interp_predict):
    "test the function to predict the mean at new sensor locations"

    mu_actual = predict_mean(A, b, fc, od, params, coords_predict)

    rho = np.exp(params[0])

    u = Function(fs)
    solve_posterior(A, u, b, fc, od, params)
    u_f = u.vector().gather()
    print(interp_predict)

    if COMM_WORLD.rank == 0:
        mu_expect = rho*np.dot(interp_predict.T, u_f)
        assert_allclose(mu_actual, mu_expect)
    else:
        assert mu_actual.shape == (0,)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_predict_covariance(fs, A, b, meshcoords, fc, od, interp, Ks, params, coords_predict, interp_predict, Ks_predict):
    "test the function to predict the mean at new sensor locations"

    Cu_actual = predict_covariance(A, b, fc, od, params, coords_predict, 0.1)

    rho = np.exp(params[0])

    full_data = ObsData(np.reshape(meshcoords, (-1, 1)), np.ones(nx + 1), 0.1)
    mu_full, Cu_full = solve_prior_covariance(A, b, fc, full_data)

    if COMM_WORLD.rank == 0:
        Cuinv = np.linalg.inv(Cu_full)
        Kinv = np.linalg.multi_dot([interp, np.linalg.inv(Ks), interp.T])
        Cu_expect = (Ks_predict +
                     rho**2*np.linalg.multi_dot([interp_predict.T, np.linalg.inv(Cuinv + rho**2*Kinv), interp_predict]))
        assert_allclose(Cu_actual, Cu_expect, atol=1.e-3, rtol=1.e-3)
    else:
        assert Cu_actual.shape == (0,0)

    Cu_actual = predict_covariance(A, b, fc, od, params, coords_predict.flatten(), 0.1)

    if COMM_WORLD.rank == 0:
        assert_allclose(Cu_actual, Cu_expect, atol=1.e-3, rtol=1.e-3)

    with pytest.raises(AssertionError):
        predict_covariance(A, b, fc, od, params, np.array([[0.2, 0.2]]), 0.1)
