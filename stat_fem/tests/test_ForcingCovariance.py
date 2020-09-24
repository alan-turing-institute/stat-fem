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
from .helper_funcs import nx, my_ensemble, comm, mesh, fs, meshcoords, fc, cov

import gc
gc.disable()

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_init(fs):
    "test init method of ForcingCovariance"

    # note: tests only handle case of a single process
    # need to think about testing more complicated cases

    # simple example in 1D

    sigma = np.log(1.)
    l = np.log(0.1)

    fc = ForcingCovariance(fs, sigma, l)

    n = Function(fs).vector().size()
    n_local = Function(fs).vector().local_size()

    M = PETSc.Mat().create()
    M.setSizes(((n_local, -1), (n_local, -1)))
    M.setFromOptions()
    M.setUp()
    start, end = M.getOwnershipRange()

    assert fc.nx == n
    assert fc.nx_local == n_local
    assert fc.function_space == fs
    assert_allclose(fc.sigma, sigma)
    assert_allclose(fc.l, l)
    assert_allclose(fc.cutoff, 1.e-3)
    assert_allclose(fc.regularization, 1.e-8)
    assert fc.local_startind == start
    assert fc.local_endind == end
    assert not fc.is_assembled

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_init_failures(fs):
    "situations where ForcingCovariance will fail"

    # bad types for inputs

    sigma = np.log(1.)
    l = np.log(0.1)

    with pytest.raises(TypeError):
        ForcingCovariance(1., sigma, l)

    # check that failure occurs with negative regularization

    with pytest.raises(AssertionError):
        ForcingCovariance(fs, sigma, l, regularization=-1.)

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_integrate_basis_functions(meshcoords, fc):
    "test the method to integrate basis functions in ForcingCovariance"

    basis_ordered = np.array([0.05, 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.05])
    basis_expected = np.zeros(11)

    meshcoords_ordered = np.linspace(0., 1., 11)

    for i in range(11):
        basis_expected[np.where(meshcoords == meshcoords_ordered[i])] = basis_ordered[i]

    assert_allclose(basis_expected, fc._integrate_basis_functions())

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_compute_G_vals(fc, cov):
    "test the compute_G_vals method of Forcing Covariance"

    nnz_expected = nx + 1

    G_dict, nnz = fc._compute_G_vals()

    assert nnz == nnz_expected

    for key, val in G_dict.items():
        assert_allclose(val[0], cov[key, val[1]], atol = 1.e-8, rtol = 1.e-6)

    assert len(G_dict) == fc.local_endind - fc.local_startind

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_assemble(fc, cov):
    "test the generate_G method of Forcing Covariance"

    fc.assemble()

    assert fc.is_assembled

    for i in range(fc.local_startind, fc.local_endind):
        for j in range(0, nx + 1):
            assert_allclose(fc.G.getValue(i, j), cov[i, j], atol = 1.e-8, rtol = 1.e-6)

    fc.destroy()

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_mult(fs, fc, cov):
    "test the multiplication method of ForcingCovariance"

    fc.assemble()

    x = Function(fs).vector()
    x.set_local(np.ones(x.local_size()))

    y = Function(fs).vector()

    fc.mult(x, y)

    ygathered = y.gather()

    assert_allclose(ygathered, np.dot(cov, np.ones(nx + 1)))

@pytest.mark.mpi
@pytest.mark.parametrize("my_ensemble", [1, 2], indirect=["my_ensemble"])
def test_ForcingCovariance_mult_parallel(my_ensemble, fs, fc, cov):
    "test that the multiplication method of ForcingCovariance can be called independently in an ensemble"

    fc.assemble()

    if my_ensemble.ensemble_comm.rank == 0:

        x = Function(fs).vector()
        x.set_local(np.ones(x.local_size()))

        y = Function(fs).vector()

        fc.mult(x, y)

        ygathered = y.gather()

        assert_allclose(ygathered, np.dot(cov, np.ones(nx + 1)))

    elif my_ensemble.ensemble_comm.rank == 1:

        x = Function(fs).vector()
        x.set_local(0.5*np.ones(x.local_size()))

        y = Function(fs).vector()

        fc.mult(x, y)

        ygathered = y.gather()

        assert_allclose(ygathered, np.dot(cov, 0.5*np.ones(nx + 1)))

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_get_nx(fc):
    "test the get_nx method of ForcingCovariance"

    assert fc.get_nx() == nx + 1

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_get_nx_local(fs, fc):
    "test the get_nx_local method of ForcingCovariance"

    n_local = Function(fs).vector().local_size()

    assert fc.get_nx_local() == n_local

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_ForcingCovariance_str(fc):
    "test the string method of ForcingCovariance"

    assert str(fc) == "Forcing Covariance with {} mesh points".format(nx + 1)

gc.collect()
