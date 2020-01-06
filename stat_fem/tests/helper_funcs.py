import numpy as np
from scipy.spatial.distance import cdist
from firedrake import UnitIntervalMesh, FunctionSpace, dx, TrialFunction, TestFunction, interpolate
from firedrake import dot, assemble, DirichletBC, grad, Function, VectorFunctionSpace
from firedrake import COMM_WORLD, Ensemble, SpatialCoordinate, pi, sin
from ..ForcingCovariance import ForcingCovariance
from ..ObsData import ObsData

def create_assembled_problem(nx, comm):
    "common firedrake problem for tests"

    mesh = UnitIntervalMesh(nx, comm=comm)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate((4.*pi*pi)*sin(x[0]*pi*2))
    a = (dot(grad(v), grad(u))) * dx
    L = f * v * dx
    bc = DirichletBC(V, 0., "on_boundary")
    A = assemble(a, bcs = bc)
    b = assemble(L)

    return A, b, mesh, V

def create_meshcoords(mesh, V):
    "shared routine for creating fem coordinates"

    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    meshcoords = X.vector().gather()

    return meshcoords

def create_forcing_covariance(mesh, V):
    "common forcing covariance object and matrix for tests"

    nx = 10

    meshcoords = create_meshcoords(mesh, V)

    sigma = np.log(1.)
    l = np.log(0.1)
    cutoff = 0.
    regularization = 1.e-8

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)
    basis = fc._integrate_basis_functions()

    r = cdist(np.reshape(meshcoords, (-1, 1)), np.reshape(meshcoords, (-1, 1)))
    cov = (np.outer(basis, basis)*np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2) +
           np.eye(nx + 1)*regularization)

    fc.assemble()

    return fc, cov

def create_obs_data():
    "create observational data object used in tests"

    coords = np.array([[0.75], [0.5], [0.25], [0.125]])
    data = np.array([-1.185, -0.06286, 0.8934, 0.7962])
    unc = 0.1
    od = ObsData(coords, data, unc)

    return od

def create_problem_numpy(mesh, V):
    "create assembled FEM problem but as a numpy array"

    nx = 10

    meshcoords = create_meshcoords(mesh, V)

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
        for j in range(nx + 1):
            ab[np.where(meshcoords == meshcoords_ordered[i]),
               np.where(meshcoords == meshcoords_ordered[j])] = ab_ordered[i, j]

    return ab, b_actual

def create_interp(mesh, V):
    "create common interpolation matrix"

    nx = 10
    nd = 4

    meshcoords = create_meshcoords(mesh, V)

    interp_ordered = np.transpose(np.array([[0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.],
                                            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                            [0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0.75, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.]]))
    interp = np.zeros((nx + 1, nd))

    meshcoords_ordered = np.linspace(0., 1., nx + 1)

    for i in range(nx + 1):
        interp[np.where(meshcoords == meshcoords_ordered[i]),:] = interp_ordered[i,:]

    return interp

def create_K_plus_sigma(sigma, l):
    "create shared model discrepancy matrix with measurement error"

    coords = np.array([[0.75], [0.5], [0.25], [0.125]])
    unc = 0.1

    r = cdist(coords, coords)
    K = np.exp(sigma)**2*np.exp(-0.5*r**2/np.exp(l)**2)+np.eye(len(coords))*unc**2

    return K

def create_interp_2(mesh, V):
    "create common interpolation matrix"

    nx = 10
    nd = 5

    meshcoords = create_meshcoords(mesh, V)

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