import numpy as np
from numpy.testing import assert_allclose
from firedrake.utility_meshes import UnitIntervalMesh
from firedrake.functionspace import FunctionSpace
import pytest
from ..ForcingCovariance import ForcingCovariance

def test_ForcingCovariance_init():
    "test init method of ForcingCovariance"

    # note: tests only handle case of a single process
    # need to think about testing more complicated cases

    # simple example in 1D

    nx = 2

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1

    fc = ForcingCovariance(V, sigma, l)

    assert fc.nx == 3
    assert fc.function_space == V
    assert_allclose(fc.sigma, sigma)
    assert_allclose(fc.l, l)
    assert_allclose(fc.cutoff, 1.e-3)
    assert_allclose(fc.regularization, 1.e-8)
    assert fc.local_startind == 0
    assert fc.local_endind == 3

    # set cutoff and regularization

    cutoff = 1.e-5
    regularization = 1.e-2

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    assert fc.nx == 3
    assert fc.function_space == V
    assert_allclose(fc.sigma, sigma)
    assert_allclose(fc.l, l)
    assert_allclose(fc.cutoff, cutoff)
    assert_allclose(fc.regularization, regularization)
    assert fc.local_startind == 0
    assert fc.local_endind == 3

def test_ForcingCovariance_init_failures():
    "situations where ForcingCovariance will fail"

    # bad types for inputs

    nx = 2

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1

    meshvals = np.array([0., 0.5, 1.])
    basis = np.array([0.25, 0.5, 0.25])

    with pytest.raises(TypeError):
        ForcingCovariance(1., sigma, l)

    # check that failure occurs with negative regularization

    with pytest.raises(AssertionError):
        ForcingCovariance(V, sigma, l, regularization=-1.)

def test_ForcingCovariance_integrate_basis_functions():
    "test the method to integrate basis functions in ForcingCovariance"

    nx = 2

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1

    fc = ForcingCovariance(V, sigma, l)

    basis = np.array([0.25, 0.5, 0.25])

    assert_allclose(basis, fc._integrate_basis_functions())

def test_ForcingCovariance_compute_G_vals():
    "test the compute_G_vals method of Forcing Covariance"

    nx = 2

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    meshvals = np.array([0., 0.5, 1.])
    basis = np.array([0.25, 0.5, 0.25])

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    r = np.array([[0., 0.5, 1.], [0.5, 0., 0.5], [1., 0.5, 0.]])
    cov_expected = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
                    np.eye(nx + 1)*regularization)
    nnz_expected = [3, 2, 1]

    G_dict, nnz = fc._compute_G_vals()

    assert nnz == nnz_expected

    for i in range(nx + 1):
        for j in range(i, nx + 1):
            assert_allclose(G_dict[(i,j)], cov_expected[i, j], atol = 1.e-8, rtol = 1.e-6)

def test_ForcingCovariance_generate_G():
    "test the generate_G method of Forcing Covariance"

    nx = 2

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    meshvals = np.array([0., 0.5, 1.])
    basis = np.array([0.25, 0.5, 0.25])

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    r = np.array([[0., 0.5, 1.], [0.5, 0., 0.5], [1., 0.5, 0.]])
    cov_expected = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
                    np.eye(nx + 1)*regularization)

    fc._generate_G()
    fc.G.assemble()

    for i in range(nx + 1):
        for j in range(i, nx + 1):
            assert_allclose(fc.G.getValue(i, j), cov_expected[i, j], atol = 1.e-8, rtol = 1.e-6)

def test_ForcingCovariance_assemble():
    "test the generate_G method of Forcing Covariance"

    nx = 2

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    meshvals = np.array([0., 0.5, 1.])
    basis = np.array([0.25, 0.5, 0.25])

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    r = np.array([[0., 0.5, 1.], [0.5, 0., 0.5], [1., 0.5, 0.]])
    cov_expected = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
                    np.eye(nx + 1)*regularization)

    fc.assemble()

    for i in range(nx + 1):
        for j in range(i, nx + 1):
            assert_allclose(fc.G.getValue(i, j), cov_expected[i, j], atol = 1.e-8, rtol = 1.e-6)
