import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from firedrake import UnitIntervalMesh, FunctionSpace, dx, TrialFunction, TestFunction, interpolate
from firedrake import dot, assemble, DirichletBC, grad, Function, VectorFunctionSpace
from firedrake import COMM_WORLD, Ensemble, SpatialCoordinate, pi, sin, solve
from ..solving import solve_posterior, solve_posterior_covariance, solve_prior_covariance
from ..ForcingCovariance import ForcingCovariance
from ..ObsData import ObsData

def test_solve_posterior():
    "test solve_conditioned_FEM"

    pass

def test_solve_posterior_covariance():
    "test solve_conditioned_FEM"

    pass

def test_solve_prior_covariance():
    "test solve_conditioned_FEM"

    nx = 10

    mesh = UnitIntervalMesh(nx)
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

    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    meshcoords = X.vector().gather()

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)
    basis = fc._integrate_basis_functions()

    fc.assemble()

    r = cdist(np.reshape(meshcoords, (-1, 1)), np.reshape(meshcoords, (-1, 1)))
    cov = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
           np.eye(nx + 1)*regularization)

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

    coords = np.array([[0.75], [0.5], [0.25], [0.125]])
    data = np.array([2., 3., 4., 5.])
    unc = 0.1
    nd = 4
    od = ObsData(coords, data, unc)
    interp_ordered = np.transpose(np.array([[0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.],
                                            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                            [0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0.75, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.]]))
    interp = np.zeros((nx + 1, nd))

    for i in range(nx + 1):
        b_actual[np.where(meshcoords == meshcoords_ordered[i])] = b_ordered[i]
        interp[np.where(meshcoords == meshcoords_ordered[i]),:] = interp_ordered[i,:]
        for j in range(nx + 1):
            ab[np.where(meshcoords == meshcoords_ordered[i]),
               np.where(meshcoords == meshcoords_ordered[j])] = ab_ordered[i, j]

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3))

    C_expected = np.linalg.solve(ab, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(ab, C_expected)
    C_expected = np.dot(interp.T, C_expected)

    u = Function(V)
    solve(A, u, b)
    m_expected = np.dot(interp.T, u.vector().gather())

    if COMM_WORLD.rank == 0:
        assert_allclose(m_expected, mu, atol = 1.e-10)
        assert_allclose(C_expected, Cu, atol = 1.e-10)
    else:
        assert mu.shape == (0,)
        assert Cu.shape == (0,0)

@pytest.mark.mpi
def test_solve_prior_covariance_parallel():
    "test solve_conditioned_FEM"

    nx = 10

    my_ensemble = Ensemble(COMM_WORLD, 2)

    mesh = UnitIntervalMesh(nx, comm=my_ensemble.comm)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate((2.*pi)*sin(x[0]*pi*2))
    a = (dot(grad(v), grad(u))) * dx
    L = f * v * dx
    bc = DirichletBC(V, 0., "on_boundary")
    A = assemble(a, bcs = bc)
    b = assemble(L)

    sigma = 1.
    l = 0.1
    cutoff = 0.
    regularization = 1.e-8

    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    meshcoords = X.vector().gather()

    fc = ForcingCovariance(V, sigma, l, cutoff, regularization)
    basis = fc._integrate_basis_functions()

    fc.assemble()

    r = cdist(np.reshape(meshcoords, (-1, 1)), np.reshape(meshcoords, (-1, 1)))
    cov = (np.outer(basis, basis)*sigma**2*np.exp(-0.5*r**2/l**2) +
           np.eye(nx + 1)*regularization)

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

    coords = np.array([[0.75], [0.5], [0.25], [0.125]])
    data = np.array([2., 3., 4., 5.])
    unc = 0.1
    nd = 4
    od = ObsData(coords, data, unc)
    interp_ordered = np.transpose(np.array([[0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.],
                                            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                            [0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0.75, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.]]))
    interp = np.zeros((nx + 1, nd))

    for i in range(nx + 1):
        b_actual[np.where(meshcoords == meshcoords_ordered[i])] = b_ordered[i]
        interp[np.where(meshcoords == meshcoords_ordered[i]),:] = interp_ordered[i,:]
        for j in range(nx + 1):
            ab[np.where(meshcoords == meshcoords_ordered[i]),
               np.where(meshcoords == meshcoords_ordered[j])] = ab_ordered[i, j]

    mu, Cu = solve_prior_covariance(A, b, fc, od, np.ones(3), my_ensemble.ensemble_comm)

    C_expected = np.linalg.solve(ab, interp)
    C_expected = np.dot(cov, C_expected)
    C_expected = np.linalg.solve(ab, C_expected)
    C_expected = np.dot(interp.T, C_expected)

    u = Function(V)
    solve(A, u, b)
    m_expected = np.dot(interp.T, u.vector().gather())

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        assert_allclose(m_expected, mu, atol = 1.e-10)
        assert_allclose(C_expected, Cu, atol = 1.e-10)
    else:
        assert mu.shape == (0,)
        assert Cu.shape == (0,0)