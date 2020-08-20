import numpy as np
from firedrake import UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction
from firedrake import SpatialCoordinate, dx, pi, sin, dot, grad, DirichletBC
from firedrake import assemble, Function, solve
import stat_fem
from stat_fem.covariance_functions import sqexp
import matplotlib.pyplot as plt

# Set up base FEM, which solves Poisson's equation on a square mesh

nx = 101

mesh = UnitSquareMesh(nx - 1, nx - 1)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x = SpatialCoordinate(mesh)
f.interpolate(-(8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))

a = (dot(grad(v), grad(u))) * dx
L = f * v * dx

bc = DirichletBC(V, 0., "on_boundary")

A = assemble(a, bcs = bc)

b = assemble(L)

u = Function(V)

solve(A, u, b)

# note that all parameters are on a log scale, so to set true values we take a logarithm first

sigma_f = np.log(2.e-2)
l_f = np.log(0.354)
rho = np.log(0.7)
sigma_eta = np.log(1.e-4)
l_eta = np.log(0.354)

sigma_y = 5.e-3
datagrid = 6
ndata = datagrid**2

x_data = np.zeros((ndata, 2))
count = 0
for i in range(datagrid):
    for j in range(datagrid):
        x_data[count, 0] = float(i+1)/float(datagrid + 1)
        x_data[count, 1] = float(j+1)/float(datagrid + 1)
        count += 1

y = (np.exp(rho)*np.sin(2.*np.pi*x_data[:,0])*np.sin(2.*np.pi*x_data[:,1]) +
     np.random.multivariate_normal(mean = np.zeros(ndata), cov = sqexp(x_data, x_data, sigma_eta, l_eta)) +
     np.random.normal(scale = sigma_y, size = ndata))

plt.figure()
plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
              u.vector().dat.data)
plt.colorbar()
plt.scatter(x_data[:,0], x_data[:,1], c = y, cmap="Greys_r")
plt.colorbar()
plt.title("Prior FEM solution and data")

# MLE estimation of parameters

G = stat_fem.ForcingCovariance(V, sigma_f, l_f)
G.assemble()

obs_data = stat_fem.ObsData(x_data, y, sigma_y)

ls = stat_fem.estimate_params_MAP(A, b, G, obs_data)

print("MLE parameter estimates:")
print(ls.params)
print("Actual input parameters:")
print(np.array([rho, sigma_eta, l_eta]))

# solve for posterior FEM solution conditioned on data

muy = Function(V)

ls.solve_posterior(muy)
muy2, Cuy = ls.solve_posterior_covariance()

plt.figure()
plt.tripcolor(mesh.coordinates.vector().dat.data[:,0], mesh.coordinates.vector().dat.data[:,1],
              muy.vector().dat.data)
plt.colorbar()
plt.scatter(x_data[:,0], x_data[:,1], c = np.diag(Cuy), cmap="Greys_r")
plt.colorbar()
plt.title("Posterior FEM solution and uncertainty")

plt.show()