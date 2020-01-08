import numpy as np
from numpy.testing import assert_allclose
import pytest
from firedrake import COMM_WORLD
from mpi4py import MPI
from ..solving import solve_prior_covariance
from ..estimation import model_loglikelihood, model_loglikelihood_deriv, estimate_params_MLE
from .helper_funcs import create_assembled_problem, create_interp
from .helper_funcs import create_obs_data, create_forcing_covariance, create_K_plus_sigma
from .helper_funcs import mesh, fs, A, b

def test_model_loglikelihood(mesh, fs, A, b):
    "test the loglikelihood method"

    fc, cov = create_forcing_covariance(mesh, fs)

    od = create_obs_data()

    params = np.zeros(3)

    mu, Cu = solve_prior_covariance(A, b, fc, od)

    loglike_actual = model_loglikelihood(params, mu, Cu, od)

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

def test_model_loglikelihood_deriv(mesh, fs, A, b):
    "test the model loglikelihood using finite differences"

    dx = 1.e-8

    fc, cov = create_forcing_covariance(mesh, fs)

    od = create_obs_data()

    params = np.zeros(3)

    mu, Cu = solve_prior_covariance(A, b, fc, od)

    loglike_deriv_actual = model_loglikelihood_deriv(params, mu, Cu, od)

    loglike_deriv_fd = np.zeros(3)

    loglike_deriv_fd[0] = (model_loglikelihood(np.array([dx, 0., 0.]), mu, Cu, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), mu, Cu, od))/dx
    loglike_deriv_fd[1] = (model_loglikelihood(np.array([0., dx, 0.]), mu, Cu, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), mu, Cu, od))/dx
    loglike_deriv_fd[2] = (model_loglikelihood(np.array([0., 0., dx]), mu, Cu, od) -
                           model_loglikelihood(np.array([0., 0., 0.]), mu, Cu, od))/dx

    assert_allclose(loglike_deriv_actual, loglike_deriv_fd, atol=1.e-5, rtol=1.e-5)

def test_estimate_params_MLE(mesh, fs, A, b):
    "test the function to use MLE to estimate parameters"

    # fixed starting point

    fc, cov = create_forcing_covariance(mesh, fs)

    od = create_obs_data()

    result = estimate_params_MLE(A, b, fc, od, start=np.zeros(3))

    root_result = COMM_WORLD.bcast(result, root=0)

    same_result = (np.allclose(root_result[0], result[0]) and np.allclose(root_result[1], result[1]))

    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not bool(diff_arg)

    # random starting point

    fc, cov = create_forcing_covariance(mesh, fs)

    od = create_obs_data()

    np.random.seed(234)
    result = estimate_params_MLE(A, b, fc, od, start=None)

    root_result = COMM_WORLD.bcast(result, root=0)

    same_result = (np.allclose(root_result[0], result[0]) and np.allclose(root_result[1], result[1]))

    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not bool(diff_arg)
