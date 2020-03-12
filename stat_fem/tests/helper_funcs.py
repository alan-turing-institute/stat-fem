import pytest
import numpy as np
from scipy.spatial.distance import cdist
from firedrake import UnitIntervalMesh, FunctionSpace, dx, TrialFunction, TestFunction, interpolate
from firedrake import dot, assemble, DirichletBC, grad, Function, VectorFunctionSpace
from firedrake import COMM_WORLD, Ensemble, SpatialCoordinate, pi, sin
from ..ForcingCovariance import ForcingCovariance
from ..ObsData import ObsData

nx = 10

@pytest.fixture
def my_ensemble(request):
    n_proc = request.param
    return Ensemble(COMM_WORLD, n_proc)

@pytest.fixture
def comm(my_ensemble):
    return my_ensemble.comm

@pytest.fixture
def mesh(comm):
    return UnitIntervalMesh(nx, comm=comm)

@pytest.fixture
def fs(mesh):
    return FunctionSpace(mesh, "CG", 1)

@pytest.fixture
def A(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    a = (dot(grad(v), grad(u))) * dx
    bc = DirichletBC(fs, 0., "on_boundary")
    A = assemble(a, bcs = bc)

    return A

@pytest.fixture
def b(mesh, fs):
    v = TestFunction(fs)
    f = Function(fs)
    x = SpatialCoordinate(mesh)
    f.interpolate((4.*pi*pi)*sin(x[0]*pi*2))
    L = f * v * dx
    b = assemble(L)

    return b

@pytest.fixture
def meshcoords(mesh, fs):
    "shared routine for creating fem coordinates"

    W = VectorFunctionSpace(mesh, fs.ufl_element())
    X = interpolate(mesh.coordinates, W)
    meshcoords = X.vector().gather()

    return meshcoords

@pytest.fixture
def fc(fs):
    sigma = np.log(1.)
    l = np.log(0.1)
    cutoff = 0.
    regularization = 1.e-8

    fc = ForcingCovariance(fs, sigma, l, cutoff, regularization)

    return fc

@pytest.fixture
def cov(fs, fc, meshcoords):
    sigma = fc.sigma
    l = fc.l
    regularization = fc.regularization
    basis = fc._integrate_basis_functions()

    r = cdist(np.reshape(meshcoords, (-1, 1)), np.reshape(meshcoords, (-1, 1)))
    cov = (np.outer(basis, basis)*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2) +
           np.eye(nx + 1)*regularization)

    return cov

@pytest.fixture
def coords(request):
    if request.param == 1:
        return np.array([[0.75], [0.5], [0.25], [0.125]])
    else:
        return np.array([[0.75], [0.5], [0.25], [0.125], [0.625]])

@pytest.fixture
def od(coords):
    "create observational data object used in tests"

    if len(coords) == 4:
        data = np.array([-1.185, -0.06286, 0.8934, 0.7962])
    else:
        data = np.array([-1.185, -0.06286, 0.8934, 0.7962, 0.])
    unc = 0.1
    od = ObsData(coords, data, unc)

    return od

@pytest.fixture
def A_numpy(meshcoords):
    "create assembled FEM problem but as a numpy array"

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

    return ab

@pytest.fixture
def b_numpy(meshcoords):

    b_ordered = np.array([ 3.8674719419474740e-01,  2.1727588820397470e+00,
                           3.5155977204985351e+00,  3.5155977204985343e+00,
                           2.1727588820397470e+00, -2.7755575615628914e-16,
                          -2.1727588820397479e+00, -3.5155977204985338e+00,
                          -3.5155977204985334e+00, -2.1727588820397470e+00,
                          -3.8674719419474768e-01])
    b_actual = np.zeros(nx + 1)

    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        b_actual[np.where(meshcoords == meshcoords_ordered[i])] = b_ordered[i]

    return b_actual

@pytest.fixture
def interp(meshcoords, coords):
    "create common interpolation matrix"

    if len(coords) == 4:
        nd = 4
        interp_ordered = np.transpose(np.array([[0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.],
                                                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                                [0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0.75, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.]]))
    else:
        nd = 5
        interp_ordered = np.transpose(np.array([[0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.],
                                                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                                [0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0.75, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 0.75, 0.25, 0., 0., 0.]]))
    interp = np.zeros((nx + 1, nd))

    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        interp[np.where(meshcoords == meshcoords_ordered[i]),:] = interp_ordered[i,:]

    return interp

@pytest.fixture
def coords_predict():
    return np.array([[0.05], [0.825], [0.45], [0.225], [0.775]])

@pytest.fixture
def interp_predict(meshcoords, coords_predict):

    nd = 5
    interp_ordered = np.transpose(np.array([[0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 0.75, 0.25, 0.],
                                            [0., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0.],
                                            [0., 0., 0.75, 0.25, 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0.25, 0.75, 0., 0.]]))
    interp = np.zeros((nx + 1, nd))

    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        interp[np.where(meshcoords == meshcoords_ordered[i]),:] = interp_ordered[i,:]

    return interp

@pytest.fixture
def params():
    return np.zeros(3)

@pytest.fixture
def Ks(od, params):
    "create shared model discrepancy matrix with measurement error"

    sigma = params[1]
    l = params[2]

    coords = od.get_coords()
    unc = od.get_unc()

    r = cdist(coords, coords)
    K = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)+np.eye(len(coords))*unc**2

    return K

@pytest.fixture
def K(coords, params):
    "create shared model discrepancy matrix with measurement error"

    sigma = params[1]
    l = params[2]

    r = cdist(coords, coords)
    K = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)

    return K

@pytest.fixture
def Ks_predict(coords_predict, params):
    "create shared model discrepancy matrix with measurement error"

    sigma = params[1]
    l = params[2]

    unc = 0.1

    r = cdist(coords_predict, coords_predict)
    K = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)+np.eye(len(coords_predict))*unc**2

    return K