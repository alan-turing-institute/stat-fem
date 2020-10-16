.. _overview:

Overview
========

The Finite Element Method (FEM) provides a powerful set of tools to enable the
approximate solution to partial differential equations by approximating
the solution as the solution to a set of algebraic equations. It is a
powerful method of analysis in a wide range of scientific fields. However,
the computational methods of the classic FEM do not lend itself well
to integrating observational data into the solution method, and instead
scientists usually resort to less formal methods such as trial and error
in order to find agreement between data and models.

An alternative approach is to re-formulate the FEM problem as a Bayesian
inference problem, and then condition the FEM solution on the observational
data at known points. This apprach is referred to as the Statistical Finite
Element Method, and the ``stat-fem`` library provides a software library
to carry out these inference problems.

Finite Element Method
---------------------

The Finite Element Method is a powerful technique in science and engineering
that can be used to compute the numerical solution to partial differential
equations. It recasts the PDE into what is known as the weak form, which allows
the solution to be written in terms of a sum over a set of basis functions
that approximate the solution in a number of small discretized subdomains.
The approximate solutions are found by solving a system of algebraic equations
that represent the solution over the discretized subdomains, and numerous
finite element software packages exist to carry out these solutions.

One challenge with using the FEM (and many other numerical solution techniques
for that matter) is that it does not easily permit integration of observations
into the solution method. FEM methods require solving a system of algebraic
equations for the unknown solution values over the discretized mesh locations,
but if we have external data that provides independent information on what
the solution value should be at those locations, the FEM method does not
have a good method for enforcing those constraints. The Statistical FEM
solves this problem by providing a robust way to combine data and
FEM models in a Bayesian framework.

Statistical Finite Element Method
---------------------------------

The Statistical FEM treats the solution to the FEM problem as a Bayesian
Inference problem. We have some prior knowledge, represented by
the solution to the FEM model before looking at the data, and we would
like to compute the posterior probability distribution of the
solution conditioned on any observed data, taking all uncertainties
in the problem into account. Note that this differs from most numerical
solution techniques: rather than having a single "solution" we instead
express our solution as a probability distribution and would
instead compute statistics such as the mean and variance to characterize
it and use it to make predictions or estimate credible intervals. As with
many applications in statistics, the method assumes that the solution
is described by a multivariate normal distribution, which means
that the posterior solution can be computed in an analytic form.

The Statistical FEM considers three sources of uncertainty in the
calculations:

1. Uncertainty in the FEM forcing, which is represented as a multivariate
   normal distribution with zero mean and a known variance and spatial
   correlation scale. This is presumed to be known, as the forcing can
   in principle be measured and its errors can be quantified.
2. Observation errors, which are considered to be independent statistical
   errors associated with measurement.
3. Model discrepancy, which assesses how well the model matches reality.
   This can be taken as known (if some prior information is known about
   how well the model is expected match reality), or can be estimated
   from the data. The model discrepancy is also assumed to be a
   multivariate normal distribution with a known variance and spatial
   correlation scale, but also includes a multiplicative scaling factor
   between the FEM solution and the data.

All three sources of error need to be weighted appropriately in order to
correctly propagate uncertainties from the inputs to the distribution
of the posterior solution.

To compute the posterior solution, an additional series of FEM solves
must be done to determine the mean and covariance (roughly 2
additional solves are needed for every sensor reading), which is the
main computational expense associated with performing inference with
this method. In some cases, the mean posterior solution is all that
is desired (in which case, the solution can be computed over the full
FEM mesh), while in other cases the uncertainties are needed. Because
the solution is a multivariate normal distribution and will in most
cases have a dense covariance matrix, this means that the full posterior
distribution cannot be computed in many cases, and is instead computed
for a limited number of spatial locations. However, this is mainly
a restriction associated with needing to store the matrices, so if
only the variances are desired (i.e. the diagonal elements of the
covariance matrix), this can be computed in principle by solving
for the full covariance in batches and only storing the diagonal
elements of the matrix.

Once the posterior solution is computed, predictions can be made
for other spatial locations. The predicted mean is just the posterior
mean evaluated at the desired location, so this is obtained for "free"
when solving the posterior and no additional computation is required.
Computing the covariance matrix at new points however does require
doing an additional set of FEM solves, with the number scaling with
the number of new points where predictions are desired.

Next, we describe the :ref:`installation` procedure and provide an example
:ref:`tutorial` to illustrate in more detail how this process works.
