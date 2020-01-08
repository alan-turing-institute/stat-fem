import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..assemble import assemble
from ..ForcingCovariance import ForcingCovariance
from ..InterpolationMatrix import InterpolationMatrix
from firedrake.matrix import Matrix
from firedrake import dx, TrialFunction, TestFunction, interpolate
from firedrake import dot, DirichletBC, grad
from .helper_funcs import create_interp, create_assembled_problem, create_forcing_covariance
from .helper_funcs import mesh, fs

def test_assemble_ForcingCovariance(mesh, fs):
    "test assemble with a forcing covariance object"

    # assemble forcing covariance

    sigma = np.log(1.)
    l = np.log(0.1)
    cutoff = 0.
    regularization = 1.e-8

    fc = ForcingCovariance(fs, sigma, l, cutoff, regularization)

    fc_2 = assemble(fc)

    assert fc_2.is_assembled

    _, cov_expected = create_forcing_covariance(mesh, fs)

    for i in range(fc_2.local_startind, fc_2.local_endind):
        for j in range(0, 11):
            assert_allclose(fc_2.G.getValue(i, j), cov_expected[i, j], atol = 1.e-8, rtol = 1.e-6)

    fc.destroy()

def test_assemble_InterpolationMatrix(mesh, fs):
    "test assemble with an interpolation matrix"

    # assemble interpolation matrix

    coords = np.array([[0.75], [0.5], [0.25], [0.125]])
    nd = len(coords)

    im = InterpolationMatrix(fs, coords)

    im_2 = assemble(im)

    assert im_2.is_assembled

    interp_expected = create_interp(mesh, fs)

    imin, imax = im_2.interp.getOwnershipRange()

    for i in range(imin, imax):
        for j in range(nd):
            assert_allclose(im_2.interp.getValue(i, j), interp_expected[i,j], atol = 1.e-10)

    im.destroy()

def test_assemble_firedrake(fs):
    "test assemble with a firedrake object"

    # assemble firedrake matrix

    u = TrialFunction(fs)
    v = TestFunction(fs)
    a = (dot(grad(v), grad(u))) * dx
    bc = DirichletBC(fs, 0., "on_boundary")
    A = assemble(a, bcs = bc)

    assert isinstance(A, Matrix)