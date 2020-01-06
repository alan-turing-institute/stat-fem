import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from firedrake import COMM_SELF, COMM_WORLD
from .solving import solve_prior_covariance
from functools import partial
from mpi4py import MPI

def model_loglikelihood(params, A, b, G, data, ensemble_comm=COMM_SELF):
    "compute the negative log-likelihood for a given model, returning value on all processes"

    params = np.array(params, dtype=np.float64)
    assert params.shape == (3,), "bad shape for model discrepancy parameters"
    rho = np.exp(params[0])

    mu, Cu = solve_prior_covariance(A, b, G, data, ensemble_comm)

    # compute log-likelihood on root process

    if ensemble_comm.rank == 0 and G.comm.rank == 0:
        KCu = rho**2*Cu + data.calc_K_plus_sigma(params[1:])
        try:
            L = cho_factor(KCu)
        except LinAlgError:
            raise LinAlgError("Error attempting to factorize the covariance matrix " +
                              "in model_loglikelihood")
        invKCudata = cho_solve(L, data.get_data() - rho*mu)
        log_like = 0.5*(data.get_n_obs()*np.log(2.*np.pi) +
                        2.*np.sum(np.log(np.diag(L[0]))) +
                        np.dot(data.get_data() - rho*mu, invKCudata))
    else:
        log_like = None

    log_like = COMM_WORLD.bcast(log_like, root=0)

    assert not log_like is None, "error in broadcasting the log likelihood"

    COMM_WORLD.barrier()

    return log_like

def model_loglikelihood_deriv(params, A, b, G, data, ensemble_comm=COMM_SELF):
    "compute the gradient of the log-likelihood"

    params = np.array(params, dtype=np.float64)
    assert params.shape == (3,), "bad shape for model discrepancy parameters"
    rho = np.exp(params[0])

    mu, Cu = solve_prior_covariance(A, b, G, data, ensemble_comm)

    # compute log-likelihood on root process

    if ensemble_comm.rank == 0 and G.comm.rank == 0:
        KCu = rho**2*Cu + data.calc_K_plus_sigma(params[1:])
        try:
            L = cho_factor(KCu)
        except LinAlgError:
            raise LinAlgError("Error attempting to factorize the covariance matrix " +
                              "in model_loglikelihood")
        invKCudata = cho_solve(L, data.get_data() - rho*mu)

        K_deriv = data.calc_K_deriv(params[1:])

        deriv = np.zeros(3)

        deriv[0] = (-np.dot(mu, invKCudata) -
                    rho*np.linalg.multi_dot([invKCudata, Cu, invKCudata]) +
                    rho*np.trace(cho_solve(L, Cu)))
        for i in range(0, 2):
            deriv[i + 1] = -0.5*(np.linalg.multi_dot([invKCudata, K_deriv[i], invKCudata]) -
                                np.trace(cho_solve(L, K_deriv[i])))
    else:
        deriv = None

    deriv = COMM_WORLD.bcast(deriv, root=0)

    assert not deriv is None, "error in broadcasting the log likelihood derivative"

    COMM_WORLD.barrier()

    return deriv

def create_loglike_functions(A, b, G, data, ensemble_comm=COMM_SELF):
    "bind the provided model ingredients to the log-likelihood functions"

    return (partial(model_loglikelihood, A=A, b=b, G=G, data=data, ensemble_comm=ensemble_comm),
            partial(model_loglikelihood_deriv, A=A, b=b, G=G, data=data, ensemble_comm=ensemble_comm))

def estimate_params_MLE(A, b, G, data, ensemble_comm=COMM_SELF, start=None, **kwargs):
    "use maximum likelihood to estimate parameters using LBFGS"

    loglike_f, loglike_deriv = create_loglike_functions(A, b, G, data, ensemble_comm)

    best_param = None
    best_mll = None

    if start is None:
        start = 5.*(np.random.random(3)-0.5)
    else:
        assert np.array(start).shape == (3,), "bad shape for starting point"
    fmin_dict = minimize(loglike_f, start, method='L-BFGS-B', jac=loglike_deriv, options=kwargs)

    if best_param is None or fmin_dict['fun'] < best_mll:
        best_param = fmin_dict['x']
        best_mll = fmin_dict['fun']

    result = (best_mll, best_param)
    root_result = COMM_WORLD.bcast(result, root=0)
    same_result = (np.allclose(root_result[0], result[0]) and np.allclose(root_result[1], result[1]))
    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not diff_arg, "minimization did not produce identical results across all processes"

    COMM_WORLD.barrier()

    return best_mll, best_param