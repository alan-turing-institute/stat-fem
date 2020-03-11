import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
import pytest
from firedrake import Ensemble, COMM_WORLD
from firedrake.assemble import assemble
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.utility_meshes import UnitIntervalMesh
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.interpolation import interpolate
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction
from ufl import dx, dot, grad
from ..InterpolationMatrix import InterpolationMatrix, interpolate_cell
from .helper_funcs import nx, comm, my_ensemble, mesh, fs, A, meshcoords, fc, coords, interp, cov, A_numpy

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix(fs, coords):
    "test InterpolationMatrix with multiple processes"

    n_proc = COMM_WORLD.size

    nd = len(coords)

    vec = Function(fs).vector()

    im = InterpolationMatrix(fs, coords)

    if COMM_WORLD.rank == 0:
        gathered_sizes = (nd, nd)
    else:
        gathered_sizes = (0, 0)

    assert im.n_data == nd
    assert im.n_data_local == nd//n_proc
    assert im.n_mesh == vec.size()
    assert im.n_mesh_local == vec.local_size()
    assert_allclose(im.coords, coords)
    assert im.function_space == fs
    assert im.meshspace_vector.size() == vec.size()
    assert im.meshspace_vector.local_size() == vec.local_size()
    assert im.dataspace_distrib.getSizes() == (nd//n_proc, nd)
    assert im.dataspace_gathered.getSizes() == gathered_sizes
    assert im.interp.getSizes() == ((vec.local_size(), vec.size()), (nd//n_proc, nd))
    assert not im.is_assembled

@pytest.mark.parametrize("comm", [COMM_WORLD])
def test_InterpolationMatrix_failures(fs):
    "test situation where InterpolationMatrix should fail"

    # bad argument for functionspace

    coords = np.array([[0.5]])

    with pytest.raises(TypeError):
        InterpolationMatrix(1., coords)

    # bad shape for coords

    coords = np.array([[0.5, 0.5]])

    with pytest.raises(AssertionError):
        InterpolationMatrix(fs, coords)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix_assemble(fs, coords, interp):
    "test the assemble method of interpolation matrix"

    # simple 1D test

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)
    im.assemble()

    assert im.is_assembled

    imin, imax = im.interp.getOwnershipRange()

    for i in range(imin, imax):
        for j in range(nd):
            assert_allclose(im.interp.getValue(i, j), interp[i,j], atol = 1.e-10)

    im.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix_gather(fs, coords):
    "test the method to gather data at root process"

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)

    im.dataspace_distrib.set(5.)

    im._gather()

    assert_allclose(im.dataspace_gathered.array, 5.)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix_scatter(fs, coords):
    "test the method to scatter data to distributed vectors"

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)

    im.dataspace_gathered.set(5.)

    im._scatter()

    assert_allclose(im.dataspace_distrib.array, 5.)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix_interp_data_to_mesh(fs, coords, meshcoords):
    "test method to interpolate data to the mesh"

    # simple 1D test

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)
    im.assemble()

    if COMM_WORLD.rank == 0:
        data = np.array([2., 1., 3., 3.])
    else:
        data = np.zeros(0)

    expected_ordered = np.array([0.  , 2.25, 2.25, 1.5 , 0.  , 1.  , 0.  , 1.  , 1.  , 0.  , 0.  ])
    expected = np.zeros(nx + 1)

    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        expected[np.where(meshcoords == meshcoords_ordered[i])] = expected_ordered[i]

    out = im.interp_data_to_mesh(data)

    with out.dat.vec_ro as vec:
        imin, imax = im.interp.getOwnershipRange()
        for i in range(imin, imax):
            assert_allclose(vec.getValue(i), expected[i], atol = 1.e-10)

    # bad input size and shape

    if COMM_WORLD.rank == 0:
        in_vec = np.zeros(nd + 1)
    else:
        in_vec = np.zeros(1)

    with pytest.raises(AssertionError):
        im.interp_data_to_mesh(in_vec)

    if COMM_WORLD.rank == 0:
        in_vec = np.zeros((nd + 1, 1))
    else:
        in_vec = np.zeros((1, 1))

    with pytest.raises(AssertionError):
        im.interp_data_to_mesh(in_vec)

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix_interp_mesh_to_data(fs, coords, meshcoords):
    "test method to interpolate from distributed mesh to data gathered at root"

    # simple 1D test

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)
    im.assemble()

    input_ordered = np.array([3., 2., 7., 4., 0., 0., 2., 1., 1., 1., 5.])

    f = Function(fs).vector()

    meshcoords_ordered = np.linspace(0., 1., 11)

    with f.dat.vec as vec:
        imin, imax = vec.getOwnershipRange()
        for i in range(imin, imax):
            vec.setValue(i, input_ordered[np.where(meshcoords_ordered == meshcoords[i])])

    if COMM_WORLD.rank == 0:
        expected = np.array([1.  , 0.  , 5.5 , 3.25])
    else:
        expected = np.zeros(0)

    out = im.interp_mesh_to_data(f)

    assert_allclose(out, expected, atol = 1.e-10)

    # failure due to bad input sizes

    mesh2 = UnitIntervalMesh(12)
    V2 = FunctionSpace(mesh2, "CG", 1)

    f2 = Function(V2).vector()
    f2.set_local(np.ones(f2.local_size()))

    with pytest.raises(AssertionError):
        im.interp_mesh_to_data(f2)

    im.destroy()

@pytest.mark.parametrize("comm, coords", [(COMM_WORLD, 1)], indirect=["coords"])
def test_InterpolationMatrix_get_meshspace_column_vector(fs, coords, meshcoords):
    "test the get_meshspace_column_vector method"

    nd = len(coords)

    im = InterpolationMatrix(fs, coords)
    im.assemble()

    vec_expected_ordered = np.array([0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.])
    vec_expected = np.zeros(nx + 1)

    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        vec_expected[np.where(meshcoords == meshcoords_ordered[i])] = vec_expected_ordered[i]

    vec = im.get_meshspace_column_vector(0)

    with vec.dat.vec_ro as petsc_vec:
        imin, imax = petsc_vec.getOwnershipRange()
        for i in range(imin, imax):
            assert_allclose(petsc_vec.getValue(i), vec_expected[i])

    # index out of range

    with pytest.raises(AssertionError):
        im.get_meshspace_column_vector(5)

def test_InterpolationMatrix_str():
    "test the str method of InterpolationMatrix"

    nx = 10

    coords = np.array([[0.75], [0.5], [0.25], [0.1]])
    nd = len(coords)

    mesh = UnitIntervalMesh(nx)
    V = FunctionSpace(mesh, "CG", 1)

    im = InterpolationMatrix(V, coords)

    assert str(im) == "Interpolation matrix from %d mesh points to %d data points".format(nx + 1, nd)

def test_interpolate_cell():
    "test the interpolate function"

    # 1D

    data_coord = np.array([0.5])
    nodal_points = np.array([[0.], [1.]])

    val = interpolate_cell(data_coord, nodal_points)

    assert_allclose(val, [0.5, 0.5])

    # 2D

    data_coord = np.array([0.5, 0.5])
    nodal_points = np.array([[0., 0.], [1., 0.], [0., 1.]])

    val = interpolate_cell(data_coord, nodal_points)

    assert_allclose(val, [0., 0.5, 0.5], atol = 1.e-10)

    # 3D

    data_coord = np.array([1./3., 1./3., 1./3.])
    nodal_points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    val = interpolate_cell(data_coord, nodal_points)

    assert_allclose(val, [0., 1./3., 1./3., 1./3.], atol = 1.e-9)

    # failure: bad dimensions

    data_coord = np.array([0.5, 0.5])
    nodal_points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    with pytest.raises(AssertionError):
        interpolate_cell(data_coord, nodal_points)
