import numpy as np
from firedrake import UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction
from firedrake import SpatialCoordinate, dx, pi, sin, dot, grad, DirichletBC
from firedrake import assemble, Function, solve
import stat_fem
from stat_fem.covariance_functions import sqexp
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

# Set up base FEM, which solves Poisson's equation on a square mesh

nx = 101

mesh = UnitSquareMesh(nx - 1, nx - 1)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x = SpatialCoordinate(mesh)
f.interpolate((8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))

a = (dot(grad(v), grad(u))) * dx
L = f * v * dx

bc = DirichletBC(V, 0., "on_boundary")

A = assemble(a, bcs = bc)

b = assemble(L)

u = Function(V)

solve(A, u, b)

# Create some fake data that is systematically different from the FEM solution.
# note that all parameters are on a log scale, so we take the true values
# and take the logarithm

# forcing covariance parameters (taken to be known)
sigma_f = np.log(2.e-2)
l_f = np.log(0.354)

# model discrepancy parameters (need to be estimated)
rho = np.log(0.7)
sigma_eta = np.log(1.e-2)
l_eta = np.log(0.5)

# data statistical errors (taken to be known)
sigma_y = 2.e-3
datagrid = 6
ndata = datagrid**2

# create fake data on a grid
x_data = np.zeros((ndata, 2))
count = 0
for i in range(datagrid):
    for j in range(datagrid):
        x_data[count, 0] = float(i+1)/float(datagrid + 1)
        x_data[count, 1] = float(j+1)/float(datagrid + 1)
        count += 1

# fake data is the true FEM solution, scaled by the mismatch factor rho, with
# correlated errors added due to model/data discrepancy and uncorrelated measurement
# errors both added to the data
y = (np.exp(rho)*np.sin(2.*np.pi*x_data[:,0])*np.sin(2.*np.pi*x_data[:,1]) +
     np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
     np.random.normal(scale = sigma_y, size = ndata))

# visualize the prior FEM solution and the synthetic data

if makeplots:
    plt.figure()
    plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
                  u.vector().dat.data)
    plt.colorbar()
    plt.scatter(x_data[:,0], x_data[:,1], c = y, cmap="Greys_r")
    plt.colorbar()
    plt.title("Prior FEM solution and data")

# Begin stat-fem solution

# Compute and assemble forcing covariance matrix using known correlated errors in forcing

G = stat_fem.ForcingCovariance(V, sigma_f, l_f)
G.assemble()

# combine data into an observational data object using known locations, observations,
# and known statistical errors

obs_data = stat_fem.ObsData(x_data, y, sigma_y)

# Use MLE (MAP with uninformative prior information) to estimate discrepancy parameters
# Should get a good estimate of these values for this example problem (if not, you
# were unlucky with random sampling!)

ls = stat_fem.estimate_params_MAP(A, b, G, obs_data)

print("MLE parameter estimates:")
print(ls.params)
print("Actual input parameters:")
print(np.array([rho, sigma_eta, l_eta]))

# Estimation function returns a re-usable LinearSolver object, which we can use to compute the
# posterior FEM solution conditioned on the data

# solve for posterior FEM solution conditioned on data

muy = Function(V)

# solve_posterior computes the full solution on the FEM grid using a Firedrake function
ls.solve_posterior(muy)

# covariance can only be computed for a select number of locations as covariance is a dense matrix
# function returns the mean/covariance as numpy arrays, not Firedrake functions
muy2, Cuy = ls.solve_posterior_covariance()

# visualize posterior FEM solution and uncertainty

if makeplots:
    plt.figure()
    plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
                  np.exp(ls.params[0])*muy.vector().dat.data)
    plt.colorbar()
    plt.scatter(x_data[:,0], x_data[:,1], c = np.diag(Cuy), cmap="Greys_r")
    plt.colorbar()
    plt.title("Posterior FEM solution and uncertainty")
    plt.show()
