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
from ufl import dx, dot, grad
import pytest
from ..ForcingCovariance import ForcingCovariance
from ..solving import _solve_forcing_covariance

def test_solve_forcing_covariance():
    "test solve_forcing_covariance"

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    bc = DirichletBC(V, 0., "on_boundary")
    A = assemble(a, bcs=bc)

    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    meshcoords = X.vector().gather()

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)
    basis = fc._integrate_basis_functions()

    fc.assemble()

    r = cdist(np.reshape(meshcoords, (-1, 1)), np.reshape(meshcoords, (-1, 1)))
    cov = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
           np.eye(nx + 1)*regularization)

    ab_ordered = np.array([[  1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                           [  0.,  20., -10.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                           [  0., -10.,  20., -10.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                           [  0.,   0., -10.,  20., -10.,   0.,   0.,   0.,   0.,   0.,   0.],
                           [  0.,   0.,   0., -10.,  20., -10.,   0.,   0.,   0.,   0.,   0.],
                           [  0.,   0.,   0.,   0., -10.,  20., -10.,   0.,   0.,   0.,   0.],
                           [  0.,   0.,   0.,   0.,   0., -10.,  20., -10.,   0.,   0.,   0.],
                           [  0.,   0.,   0.,   0.,   0.,   0., -10.,  20., -10.,   0.,   0.],
                           [  0.,   0.,   0.,   0.,   0.,   0.,   0., -10.,  20., -10.,   0.],
                           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., -10.,  20.,   0.],
                           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.]])
    ab = np.zeros((nx + 1, nx + 1))

    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        for j in range(nx + 1):
            ab[np.where(meshcoords == meshcoords_ordered[i]),
               np.where(meshcoords == meshcoords_ordered[j])] = ab_ordered[i, j]

    rhs = Function(V).vector()
    rhs.set_local(np.ones(fc.get_nx_local()))

    result = _solve_forcing_covariance(fc, A, rhs)

    result_actual = result.gather()

    result_expected = np.linalg.solve(ab, np.ones(nx + 1))
    result_expected = np.dot(cov, result_expected)
    result_expected = np.linalg.solve(ab, result_expected)

    assert_allclose(result_expected, result_actual, atol = 1.e-10)

def test_solve_conditioned_FEM():
    "test solve_conditioned_FEM"

    pass