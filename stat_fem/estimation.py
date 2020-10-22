import numpy as np
from scipy.optimize import minimize
from firedrake import COMM_SELF, COMM_WORLD
from .LinearSolver import LinearSolver
from mpi4py import MPI

def estimate_params_MAP(A, b, G, data, priors=[None, None, None], start=None, ensemble_comm=COMM_SELF, **kwargs):
    """
    Estimate model hyperparameters using MAP estimation

    This function uses maximum a posteriori estimation to fit parameters for a Statistical FEM model.
    The function is a wrapper to the Scipy LBFGS function to minimize the marginal log posterior,
    returning a fit ``LinearSolver`` object. This allows re-use of the cached Forcing Covariance
    function solves for each sensor, which greatly improves efficiency of the computation.

    Priors on the three hyperparameters can be specified by passing a list of ``Prior``-derived
    objects. The priors are over the values as passed directly to all functions (i.e. while still
    on a logarithmic scale). If no priors are provided, then an uninformative prior is assumed
    and estimation is based on the marginal log-likelihood.

    The computations to solve for the forcing covariance can be carried out in parallel by passing
    a Firedrake Ensemble communicator for the ``ensemble_comm`` keyword argument. All other
    computations are done on the root process and broadcast to all other processes to
    ensure that all processes have the same final minimum value when the computation terminates.

    :param A: FEM stiffness matrix, must be a Firedrake Matrix class.
    :type A: Matrix
    :param b: FEM RHS vector, must be a Firedrake Function or Vector
    :type b: Function or Vector
    :param G: Forcing Covariance matrix, must be a ``ForcingCovariance`` class.
    :type G: ForcingCovariance
    :param data: Observational data object, must be a ``ObsData`` class.
    :type data: ObsData
    :param priors: List of prior objects over hyperparameters. Must be a list of length 3
                   containing ``Prior``-derived objects or ``None`` for uninformative priors.
                   Optional, default is ``[None, None, None]`` (uninformative priors on all
                   parameters).
    :type priors: list
    :param start: Starting point for minimization routine. Must be a numpy array of length 3 or
                  ``None`` if the starting point is to be drawn randomly. Optional, default is
                  ``None``.
    :type start: None or ndarray
    :param ensemble_comm: MPI communicator for ensemble parallelism created from a Firedrake
                          ``Ensemble`` object. This controls how the solves over all sensors
                          are parallelized. Optional, default value is ``COMM_SELF`` indicating
                          that forcing covariance solves are not parallelized.
    :type enemble_comm: MPI Communicator
    :param kwargs: Additional keyword arguments to be passed to either the Firedrake
                   `LinearSolver` object or the Scipy `minimize` routine. See
                   the corresponding manuals for more information.
    :returns: A LinearSolver object with the hyperparameters set to the MAP/MLE value. To extract
              the actual parameters, use the ``params`` attribute.
    :rtype: LinearSolver
    """

    # extract kwargs for firedrake solver and minimize
    
    firedrake_kwargs = ["P", "solver_parameters", "nullspace",
                        "transpose_nullspace", "near_nullspace",
                        "options_prefix"]

    ls_kwargs = {}
    minimize_kwargs = {}
    
    for kw in kwargs.keys():
        if kw in firedrake_kwargs:
            ls_kwargs[kw] = kwargs[kw]
        else:
            minimize_kwargs[kw] = kwargs[kw]
    
    ls = LinearSolver(A, b, G, data, priors=priors, ensemble_comm=ensemble_comm,
                      **ls_kwargs)

    ls.solve_prior()

    if start is None:
        if COMM_WORLD.rank == 0:
            start = 5.*(np.random.random(3)-0.5)
        else:
            start = None
        start = COMM_WORLD.bcast(start, root=0)
    else:
        assert np.array(start).shape == (3,), "bad shape for starting point"

    fmin_dict = minimize(ls.logposterior, start, method='L-BFGS-B',
                         jac=ls.logpost_deriv, options=minimize_kwargs)

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
