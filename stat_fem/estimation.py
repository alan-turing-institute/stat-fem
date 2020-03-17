import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from firedrake import COMM_SELF, COMM_WORLD
from .LinearSolver import LinearSolver
from .solving import solve_prior_covariance
from functools import partial
from mpi4py import MPI

def estimate_params_MAP(A, b, G, data, priors=[None, None, None], start=None, ensemble_comm=COMM_SELF, **kwargs):
    "use maximum a posteriori to estimate parameters using LBFGS, returning a fit LinearSolver object"

    ls = LinearSolver(A, b, G, data, priors, ensemble_comm)

    ls.solve_prior()

    if start is None:
        if COMM_WORLD.rank == 0:
            start = 5.*(np.random.random(3)-0.5)
        else:
            start = None
        start = COMM_WORLD.bcast(start, root=0)
    else:
        assert np.array(start).shape == (3,), "bad shape for starting point"

    fmin_dict = minimize(ls.logposterior, start, method='L-BFGS-B', jac=ls.logpost_deriv, options=kwargs)

    assert fmin_dict['success'], "minimization routine failed"

    # broadcast result to all processes

    result = fmin_dict['x']
    root_result = COMM_WORLD.bcast(result, root=0)
    same_result = np.allclose(root_result, result)
    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not diff_arg, "minimization did not produce identical results across all processes"

    COMM_WORLD.barrier()

    ls.set_params(root_result)

    return ls