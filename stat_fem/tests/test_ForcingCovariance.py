import numpy as np
from scipy.spatial.distance import cdist
from numpy.testing import assert_allclose
from firedrake import COMM_WORLD
from firedrake.ensemble import Ensemble
from firedrake.function import Function
from firedrake.utility_meshes import UnitIntervalMesh
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.interpolation import interpolate
from firedrake.petsc import PETSc
import pytest
from ..ForcingCovariance import ForcingCovariance

def test_ForcingCovariance_init():
    "test init method of ForcingCovariance"

    # note: tests only handle case of a single process
    # need to think about testing more complicated cases

    # simple example in 1D

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1

    fc = ForcingCovariance(V, sigma, l)

    n = Function(V).vector().size()
    n_local = Function(V).vector().local_size()

    M = PETSc.Mat().create()
    M.setSizes(((n_local, -1), (n_local, -1)))
    M.setFromOptions()
    M.setUp()
    start, end = M.getOwnershipRange()

    assert fc.nx == n
    assert fc.nx_local == n_local
    assert fc.function_space == V
    assert_allclose(fc.sigma, sigma)
    assert_allclose(fc.l, l)
    assert_allclose(fc.cutoff, 1.e-3)
    assert_allclose(fc.regularization, 1.e-8)
    assert fc.local_startind == start
    assert fc.local_endind == end

    # set cutoff, regularization, and use an ensemble

    cutoff = 1.e-5
    regularization = 1.e-2

    ensemble_comm = Ensemble(COMM_WORLD, COMM_WORLD.size)

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization, comm=ensemble_comm)

    assert fc.nx == n
    assert fc.nx_local == n_local
    assert fc.function_space == V
    assert_allclose(fc.sigma, sigma)
    assert_allclose(fc.l, l)
    assert_allclose(fc.cutoff, cutoff)
    assert_allclose(fc.regularization, regularization)
    assert fc.local_startind == start
    assert fc.local_endind == end

def test_ForcingCovariance_init_failures():
    "situations where ForcingCovariance will fail"

    # bad types for inputs

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1

    with pytest.raises(TypeError):
        ForcingCovariance(1., sigma, l)

    # check that failure occurs with negative regularization

    with pytest.raises(AssertionError):
        ForcingCovariance(V, sigma, l, regularization=-1.)

    # MPI comm bad type

    with pytest.raises(TypeError):
        ForcingCovariance(V, sigma, l, comm=1.)

def test_ForcingCovariance_integrate_basis_functions():
    "test the method to integrate basis functions in ForcingCovariance"

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1

    fc = ForcingCovariance(V, sigma, l)

    basis_ordered = np.array([0.05, 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.05])
    basis_expected = np.zeros(nx + 1)

    meshcoords = V.mesh().coordinates.vector().gather()
    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        basis_expected[np.where(meshcoords == meshcoords_ordered[i])] = basis_ordered[i]

    assert_allclose(basis_expected, fc._integrate_basis_functions())

def test_ForcingCovariance_compute_G_vals():
    "test the compute_G_vals method of Forcing Covariance"

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    meshcoords = X.vector().gather()

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    basis = fc._integrate_basis_functions()

    r = cdist(np.reshape(meshcoords, (-1, 1)), np.reshape(meshcoords, (-1, 1)))
    cov_expected = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
                    np.eye(nx + 1)*regularization)
    nnz_expected = [nx + 1]*(fc.local_endind - fc.local_startind)

    G_dict, nnz = fc._compute_G_vals()

    assert nnz == nnz_expected

    for key, val in G_dict.items():
        assert_allclose(val, cov_expected[key[0], key[1]], atol = 1.e-8, rtol = 1.e-6)

    assert len(G_dict) == len(cov_expected[fc.local_startind:fc.local_endind,:].flatten())

def test_ForcingCovariance_assemble():
    "test the generate_G method of Forcing Covariance"

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    meshcoords = X.vector().gather()

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)
    basis = fc._integrate_basis_functions()

    r = cdist(np.reshape(meshcoords, (-1, 1)), np.reshape(meshcoords, (-1, 1)))
    cov_expected = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
                    np.eye(nx + 1)*regularization)

    fc.assemble()

    for i in range(fc.local_startind, fc.local_endind):
        for j in range(0, nx + 1):
            assert_allclose(fc.G.getValue(i, j), cov_expected[i, j], atol = 1.e-8, rtol = 1.e-6)

    fc.destroy()

def test_ForcingCovariance_get_nx():
    "test the get_nx method of ForcingCovariance"

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    assert fc.get_nx() == nx + 1

def test_ForcingCovariance_get_nx_local():
    "test the get_nx_local method of ForcingCovariance"

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    n_local = Function(V).vector().local_size()

    assert fc.get_nx_local() == n_local

def test_ForcingCovariance_str():
    "test the string method of ForcingCovariance"

    nx = 10

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)
    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)

    assert str(fc) == "Forcing Covariance with {} mesh points".format(nx + 1)