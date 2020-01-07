import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..assemble import assemble
from ..ForcingCovariance import ForcingCovariance
from ..InterpolationMatrix import InterpolationMatrix
from firedrake.matrix import Matrix
from firedrake import dx, TrialFunction, TestFunction, interpolate
from firedrake import dot, DirichletBC, grad, Function
from firedrake import COMM_WORLD, SpatialCoordinate, pi, sin
from .helper_funcs import create_interp, create_assembled_problem, create_forcing_covariance

def test_assemble_ForcingCovariance():
    "test assemble with a forcing covariance object"

    # assemble forcing covariance

    nx = 10

    _, _, mesh, V = create_assembled_problem(nx, COMM_WORLD)
    sigma = np.log(1.)
    l = np.log(0.1)
    cutoff = 0.
    regularization = 1.e-8

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    fc_2 = assemble(fc)

    assert fc_2.is_assembled

    _, cov_expected = create_forcing_covariance(mesh, V)

    for i in range(fc_2.local_startind, fc_2.local_endind):
        for j in range(0, nx + 1):
            assert_allclose(fc_2.G.getValue(i, j), cov_expected[i, j], atol = 1.e-8, rtol = 1.e-6)

    fc.destroy()

def test_assemble_InterpolationMatrix():
    "test assemble with an interpolation matrix"

    # assemble interpolation matrix

    nx = 10

    coords = np.array([[0.75], [0.5], [0.25], [0.125]])
    nd = len(coords)

    A, b, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    im = InterpolationMatrix(V, coords)

    im_2 = assemble(im)

    assert im_2.is_assembled

    interp_expected = create_interp(mesh, V)

    imin, imax = im_2.interp.getOwnershipRange()

    for i in range(imin, imax):
        for j in range(nd):
            assert_allclose(im_2.interp.getValue(i, j), interp_expected[i,j], atol = 1.e-10)

    im.destroy()

def test_assemble_firedrake():
    "test assemble with a firedrake object"

    # assemble firedrake matrix

    nx = 10

    _, _, mesh, V = create_assembled_problem(nx, COMM_WORLD)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate((4.*pi*pi)*sin(x[0]*pi*2))
    a = (dot(grad(v), grad(u))) * dx
    L = f * v * dx
    bc = DirichletBC(V, 0., "on_boundary")
    A = assemble(a, bcs = bc)

    assert isinstance(A, Matrix)