import numpy as np
from numpy.testing import assert_allclose
import pytest
from firedrake import COMM_WORLD
from mpi4py import MPI
from ..estimation import estimate_params_MAP
from .helper_funcs import my_ensemble, comm, mesh, fs, A, b, fc, coords, od, params, Ks

import gc
gc.disable()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_estimate_params_MAP(A, b, fc, od):
    "test the function to use MLE to estimate parameters"

    # fixed starting point

    result = estimate_params_MAP(A, b, fc, od, start=np.zeros(3))

    root_result = COMM_WORLD.bcast(result.params, root=0)

    same_result = np.allclose(root_result, result.params)

    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not bool(diff_arg)

    # random starting point

    np.random.seed(234)

    result = estimate_params_MAP(A, b, fc, od, start=None)

    root_result = COMM_WORLD.bcast(result.params, root=0)

    same_result = np.allclose(root_result, result.params)

    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not bool(diff_arg)

    # check that args are passed on to linear solver and minimize
    
    result = estimate_params_MAP(A, b, fc, od, start = np.zeros(3),
                                 solver_parameters={}, ftol=1.e-10)

    root_result = COMM_WORLD.bcast(result.params, root=0)

    same_result = np.allclose(root_result, result.params)

    diff_arg = COMM_WORLD.allreduce(int(not same_result), op=MPI.SUM)

    assert not bool(diff_arg)

gc.collect()
