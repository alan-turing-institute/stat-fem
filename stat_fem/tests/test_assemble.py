import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..assemble import assemble
from ..ForcingCovariance import ForcingCovariance
from ..InterpolationMatrix import InterpolationMatrix
from firedrake.matrix import Matrix
from firedrake import dx, TrialFunction, TestFunction, interpolate
from firedrake import dot, DirichletBC, grad, COMM_WORLD
from .helper_funcs import nx, my_ensemble, comm, mesh, fs, fc, cov, meshcoords, coords, interp

import gc
gc.disable()

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_assemble_ForcingCovariance(fc, cov):
    "test assemble with a forcing covariance object"

    # assemble forcing covariance

    fc_2 = assemble(fc)

    assert fc_2.is_assembled

    for i in range(fc_2.local_startind, fc_2.local_endind):
        for j in range(0, nx + 1):
            assert_allclose(fc_2.G.getValue(i, j), cov[i, j], atol = 1.e-8, rtol = 1.e-6)

    fc.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_assemble_InterpolationMatrix(fs, coords, interp):
    "test assemble with an interpolation matrix"

    # assemble interpolation matrix

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)

    im_2 = assemble(im)

    assert im_2.is_assembled

    imin, imax = im_2.interp.getOwnershipRange()

    for i in range(imin, imax):
        for j in range(nd):
            assert_allclose(im_2.interp.getValue(i, j), interp[i,j], atol = 1.e-10)

    im.destroy()

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_assemble_firedrake(fs):
    "test assemble with a firedrake object"

    # assemble firedrake matrix

    u = TrialFunction(fs)
    v = TestFunction(fs)
    a = (dot(grad(v), grad(u))) * dx
    bc = DirichletBC(fs, 0., "on_boundary")
    A = assemble(a, bcs = bc)

    assert isinstance(A, Matrix)

gc.collect()
