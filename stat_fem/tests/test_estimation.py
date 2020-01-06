import numpy as np
from numpy.testing import assert_allclose
import pytest
from firedrake import COMM_WORLD, Ensemble
from mpi4py import MPI
from ..solving import solve_prior_covariance
from ..estimation import model_loglikelihood, model_loglikelihood_deriv, create_loglike_functions
from ..estimation import estimate_params_MLE
from .helper_funcs import create_assembled_problem, create_interp
from .helper_funcs import create_obs_data, create_problem_numpy, create_forcing_covariance
from .helper_funcs import create_K_plus_sigma, create_meshcoords

def test_model_loglikelihood():
    "test the loglikelihood method"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    params = np.zeros(3)

    mu, Cu = solve_prior_covariance(A, b, fc, od)

    loglike_actual = model_loglikelihood(params, A, b, fc, od)

    if COMM_WORLD.rank == 0:
        KCu = Cu + od.calc_K_plus_sigma(params[1:])
        loglike_expected = 0.5*(np.linalg.multi_dot([od.get_data() - mu,
                                                     np.linalg.inv(KCu),
                                                     od.get_data() - mu]) +
                                np.log(np.linalg.det(KCu)) +
                                od.get_n_obs()*np.log(2.*np.pi))
    else:
        loglike_expected = None

    loglike_expected = COMM_WORLD.bcast(loglike_expected, root=0)

    assert_allclose(loglike_expected, loglike_actual)

@pytest.mark.mpi
@pytest.mark.parametrize("n_proc", [1, 2])
def test_model_loglikelihood_parallel(n_proc):
    "test the loglikelihood method"

    my_ensemble = Ensemble(COMM_WORLD, n_proc)

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    params = np.zeros(3)

    mu, Cu = solve_prior_covariance(A, b, fc, od)

    loglike_actual = model_loglikelihood(params, A, b, fc, od, my_ensemble.ensemble_comm)

    if COMM_WORLD.rank == 0:
        KCu = Cu + od.calc_K_plus_sigma(params[1:])
        loglike_expected = 0.5*(np.linalg.multi_dot([od.get_data() - mu,
                                                     np.linalg.inv(KCu),
                                                     od.get_data() - mu]) +
                                np.log(np.linalg.det(KCu)) +
                                od.get_n_obs()*np.log(2.*np.pi))
    else:
        loglike_expected = None

    loglike_expected = COMM_WORLD.bcast(loglike_expected, root=0)

    assert_allclose(loglike_expected, loglike_actual)

def test_model_loglikelihood_deriv():
    "test the model loglikelihood using finite differences"

    dx = 1.e-8

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    params = np.zeros(3)

    loglike_deriv_actual = model_loglikelihood_deriv(params, A, b, fc, od)

    loglike_deriv_fd = np.zeros(3)

    loglike_deriv_fd[0] = (model_loglikelihood(np.array([dx, 0., 0.]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx
    loglike_deriv_fd[1] = (model_loglikelihood(np.array([0., dx, 0.]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx
    loglike_deriv_fd[2] = (model_loglikelihood(np.array([0., 0., dx]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx

    assert_allclose(loglike_deriv_actual, loglike_deriv_fd, atol=1.e-5, rtol=1.e-5)

@pytest.mark.mpi
@pytest.mark.parametrize("n_proc", [1, 2])
def test_model_loglikelihood_deriv_parallel(n_proc):
    "test the model loglikelihood using finite differences"

    dx = 1.e-8

    my_ensemble = Ensemble(COMM_WORLD, n_proc)

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, my_ensemble.comm)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    params = np.zeros(3)

    mu, Cu = solve_prior_covariance(A, b, fc, od)

    loglike_deriv_actual = model_loglikelihood_deriv(params, A, b, fc, od, my_ensemble.ensemble_comm)

    loglike_deriv_fd = np.zeros(3)

    loglike_deriv_fd[0] = (model_loglikelihood(np.array([dx, 0., 0.]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx
    loglike_deriv_fd[1] = (model_loglikelihood(np.array([0., dx, 0.]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx
    loglike_deriv_fd[2] = (model_loglikelihood(np.array([0., 0., dx]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx

    assert_allclose(loglike_deriv_actual, loglike_deriv_fd, atol=1.e-5, rtol=1.e-5)

def test_create_loglike_functions():
    "test method to create a loglikelihood function with the model bound to it"

    dx = 1.e-8

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    params = np.zeros(3)

    mu, Cu = solve_prior_covariance(A, b, fc, od)

    loglike_f, loglike_deriv = create_loglike_functions(A, b, fc, od)

    loglike_actual = loglike_f(params)
    loglike_deriv_actual = loglike_deriv(params)

    if COMM_WORLD.rank == 0:
        KCu = Cu + od.calc_K_plus_sigma(params[1:])
        loglike_expected = 0.5*(np.linalg.multi_dot([od.get_data() - mu,
                                                     np.linalg.inv(KCu),
                                                     od.get_data() - mu]) +
                                np.log(np.linalg.det(KCu)) +
                                od.get_n_obs()*np.log(2.*np.pi))
    else:
        loglike_expected = None

    loglike_expected = COMM_WORLD.bcast(loglike_expected, root=0)

    loglike_deriv_fd = np.zeros(3)

    loglike_deriv_fd[0] = (model_loglikelihood(np.array([dx, 0., 0.]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx
    loglike_deriv_fd[1] = (model_loglikelihood(np.array([0., dx, 0.]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx
    loglike_deriv_fd[2] = (model_loglikelihood(np.array([0., 0., dx]), A, b, fc, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), A, b, fc, od))/dx

    assert_allclose(loglike_expected, loglike_actual)
    assert_allclose(loglike_deriv_fd, loglike_deriv_actual, atol=1.e-5, rtol=1.e-5)

def test_estimate_params_MLE():
    "test the function to use MLE to estimate parameters"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    od = create_obs_data()

    result = estimate_params_MLE(A, b, fc, od, start=np.zeros(3))
    #result = (0., np.array([1., 2., 3.]))

    root_result = COMM_WORLD.bcast(result, root=0)

    same_result = (np.allclose(root_result[0], result[0]) and np.allclose(root_result[1], result[1]))

    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not bool(diff_arg)
