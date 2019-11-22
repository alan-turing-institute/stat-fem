import numpy as np
from numpy.testing import assert_allclose
import pytest
from firedrake.utility_meshes import UnitIntervalMesh
from firedrake.functionspace import FunctionSpace
from ..InterpolationMatrix import InterpolationMatrix
from firedrake.function import PointNotInDomainError

def test_InterpolationMatrix():
    "test InterpolationMatrix"

    nx = 2

    coords = np.array([[0.5]])

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)

    im = InterpolationMatrix(coords, V)

    interp_expected = np.array([[0., 1., 0.]])

    for i in range(1):
        for j in range(nx + 1):
            assert_allclose(im.interp.getValue(i, j), interp_expected[i, j], atol = 1.e-8, rtol = 1.e-6)

    coords = np.array([[0.25]])

    im = InterpolationMatrix(coords, V)

    interp_expected = np.array([[0.5, 0.5, 0.]])

    for i in range(1):
        for j in range(nx + 1):
            assert_allclose(im.interp.getValue(i, j), interp_expected[i, j], atol = 1.e-8, rtol = 1.e-6)

def test_InterpolationMatrix_failures():
    "test situation where InterpolationMatrix should fail"

    # bad argument for functionspace

    coords = np.array([[0.5]])

    with pytest.raises(TypeError):
        InterpolationMatrix(coords, 1.)

    # coords out of domain

    coords = np.array([[-1.]])

    nx = 2

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)

    with pytest.raises(PointNotInDomainError):
        InterpolationMatrix(coords, V)

    # bad shape for coords

    coords = np.array([[0.5, 0.5]])

    with pytest.raises(AssertionError):
        InterpolationMatrix(coords, V)