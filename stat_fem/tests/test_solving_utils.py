import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from firedrake.assemble import assemble
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.utility_meshes import UnitIntervalMesh
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.interpolation import interpolate
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake import COMM_WORLD
from ufl import dx, dot, grad
import pytest
from ..ForcingCovariance import ForcingCovariance
from ..solving_utils import _solve_forcing_covariance
from .helper_funcs import create_assembled_problem, create_forcing_covariance, create_problem_numpy

def test_solve_forcing_covariance():
    "test solve_forcing_covariance"

    nx = 10

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    fc, cov = create_forcing_covariance(mesh, V)

    ab, _ = create_problem_numpy(mesh, V)

    rhs = Function(V).vector()
    rhs.set_local(np.ones(fc.get_nx_local()))

    result = _solve_forcing_covariance(fc, A, rhs)

    result_actual = result.gather()

    result_expected = np.linalg.solve(ab, np.ones(nx + 1))
    result_expected = np.dot(cov, result_expected)
    result_expected = np.linalg.solve(ab, result_expected)

    assert_allclose(result_expected, result_actual, atol = 1.e-10)
