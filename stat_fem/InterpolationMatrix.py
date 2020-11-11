import numpy as np
from firedrake import Function
from firedrake.petsc import PETSc
from firedrake.functionspace import VectorFunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from firedrake.interpolation import interpolate
from firedrake.vector import Vector

class InterpolationMatrix(object):
    """
    Class representing an interpolation matrix

    This class holds the sparse matrix needed to interpolate from the
    mesh to the data. It also handles much of the parallel communication
    between processes to enable serial linear algebra on the data while
    maintaining parallel FEM solves that are distributed across
    processes.

    In most cases, the user should not need to create ``InterpolationMatrix``
    objects directly -- these can be constructed from the information
    available in other objects used in the ``stat-fem`` solves.

    :ivar function_space: The FEM function space for the solution
    :type function_space: Firedrake FunctionSpace or related (there are
                          several permitted types, has only been tested
                          with the ``WithGeometry`` flavor)
    :ivar coords: Spatial locations where data observations are available.
                  Must be a 2D numpy array, where the first index is
                  the index representing the different sensor locations
                  and the second index represents the spatial dimensions.
    :type coords: ndarray
    :ivar n_data: Number of sensor locations
    :type n_data: int
    :ivar dataspace_distrib: Distributed PETSc Vector holding data items
                             over all MPI processes involved
    :type dataspace_distrib: PETSc Vec
    :ivar n_data_local: Number of data items held on the local process
    :type n_data_local: int
    :ivar dataspace_gathered: PETSc Vector with all sensor data collected
                              on the root process.
    :type dataspace_gathered: PETSc Vec
    :ivar petsc_scatter: PETSc Scatter object used to transfer data between
                         ``dataspace_gathered`` and ``dataspace_distrib``
    :type petsc_scatter: PETSc Scatter
    :ivar meshspace_vector: Firedrake Vector holding FEM mesh data
    :type meshspace_vector: Firedrake Vector
    :ivar n_mesh: Number of mesh DOFs in the Firedrake Function
    :type n_mesh: int
    :ivar n_mesh_local: Number of local mesh DOFs on the local process
    :type n_mesh_local: int
    :ivar interp: PETSc Sparse Matrix for interpolating between the data space
                  and mesh space solutions
    :type interp: PETSc Mat
    :ivar is_assembled: Boolean indicating if ``interp`` has been assembled
    :type is_assembled: bool
    """
    def __init__(self, function_space, coords):
        "create and assemble interpolation matrix"

        if not isinstance(function_space, WithGeometry):
            raise TypeError("bad input type for function_space: must be a FunctionSpace")

        self.coords = np.copy(coords)
        self.function_space = function_space
        self.comm = function_space.comm

        self.n_data = coords.shape[0]
        assert (coords.shape[1] == self.function_space.mesh().cell_dimension()
                ), "shape of coordinates does not match mesh dimension"

        # allocate working vectors to handle parallel matrix operations and data transfer

        # dataspace_vector is a distributed PETSc vector in the data space

        self.dataspace_distrib = PETSc.Vec().create(comm=self.comm)
        self.dataspace_distrib.setSizes((-1, self.n_data))
        self.dataspace_distrib.setFromOptions()

        self.n_data_local = self.dataspace_distrib.getSizes()[0]

        # all data computations are done on root process, so create gather method to
        # facilitate this data transfer

        self.petsc_scatter, self.dataspace_gathered = PETSc.Scatter.toZero(self.dataspace_distrib)

        self.meshspace_vector = Function(self.function_space).vector()

        self.n_mesh_local = self.meshspace_vector.local_size()
        self.n_mesh = self.meshspace_vector.size()

        nnz = len(self.function_space.cell_node_list[0])

        self.interp = PETSc.Mat().create(comm=self.comm)
        self.interp.setSizes(((self.n_mesh_local, -1), (self.n_data_local, -1)))
        self.interp.setPreallocationNNZ(nnz)
        self.interp.setFromOptions()
        self.interp.setUp()

        self.is_assembled = False

    def assemble(self):
        """
        Compute values and assemble interpolation matrix

        Assembly function to compute the nonzero sparse matrix entries
        and assemble the sparse matrix. Should only need to be called
        once for each analysis and thus will return if ``is_assembled``
        is ``True`` without re-assembling the matrix.
        """

        if self.is_assembled:
            return

        mesh = self.function_space.ufl_domain()
        W = VectorFunctionSpace(mesh, self.function_space.ufl_element())
        X = interpolate(mesh.coordinates, W)
        meshvals_local = np.array(X.dat.data_with_halos)
        imin, imax = self.interp.getOwnershipRange()

        # loop over all data points

        for i in range(self.n_data):
            cell = self.function_space.mesh().locate_cell(self.coords[i])
            if not cell is None:
                nodes = self.function_space.cell_node_list[cell]
                points = meshvals_local[nodes]
                interp_coords = interpolate_cell(self.coords[i], points)
                for (node, val) in zip(nodes, interp_coords):
                    if node < self.n_mesh_local:
                        self.interp.setValue(imin + node, i, val)

        self.interp.assemble()

        self.is_assembled = True

    def _gather(self):
        """
        Wrapper to transfer data from distributed dataspace vector to root

        This function provides a wrapper to the ``petsc_scatter`` object
        to move data from the distributed PETSc dataspace vector to the
        one with all data collected on the root process. Most Linear
        Algebra operations on the data are done on the root process due
        to the modest size of the data when compared to the FEM solution.
        """

        self.petsc_scatter.scatter(self.dataspace_distrib, self.dataspace_gathered,
                                   mode=PETSc.ScatterMode.SCATTER_FORWARD)

    def _scatter(self):
        """
        Wrapper to transfer data from root to distributed dataspace vector

        This function provides a wrapper to the ``petsc_scatter`` object
        to move data from the gathered PETSc dataspace vector to the
        distributed vector. Interpolation from the data to the mesh
        requires that the data vector be a distributed PETSc Vector
        to take advantage of parallelization of the FEM solves.
        """

        self.petsc_scatter.scatter(self.dataspace_gathered, self.dataspace_distrib,
                                   mode=PETSc.ScatterMode.SCATTER_REVERSE)

    def destroy(self):
        """
        Deallocate memory for PETSc vectors and matrix

        This function deallocates the PETSc objects with a PETSc-like
        interface. It will call the ``destroy`` method on all allocated
        PETSc objects within the ``InterpolationMatrix`` class.
        """

        self.dataspace_gathered.destroy()
        self.dataspace_distrib.destroy()
        self.interp.destroy()

    def interp_data_to_mesh(self, data_array):
        """
        Interpolate a vector of sensor values to the FEM mesh

        This function takes a gathered numpy array in the data space that
        is held on the root process and interpolates it to a distributed
        mesh vector. The provided sensor values are first scattered to
        a distributed PETSc Vector and then interpolation is done by
        multiplying the vector by the sparse interpolation matrix.

        The numpy array must be defined only on the root process (all other
        processes must have an array length of zero). This is checked
        against the expected sizes of the ``dataspace_gathered`` Vector
        on each process, an an assertion error will be triggered if these
        do not match.

        Returns a Firedrake Function holding the interpolated values on
        the FEM mesh.

        :param data_array: Numpy array holding the sensor values. Must be
                           a 1D array with the full data on the root process
                           and empty (length zero) arrays on all other
                           processes.
        :type data_array: ndarray
        :returns: Data interpolated to the FEM mesh DOFs.
        :rtype: Firedrake Vector
        """

        if not self.is_assembled:
            self.assemble()

        data_array = np.array(data_array)

        assert data_array.ndim == 1, "input data must be a 1D array of sensor values"
        assert data_array.shape[0] == self.dataspace_gathered.getSizes()[0], "data_array has an incorrect number of sensor measurements"

        # scatter into dataspace_distrib

        self.dataspace_gathered.array = np.copy(data_array)
        self._scatter()

        with self.meshspace_vector.dat.vec as vec:
            self.interp.mult(self.dataspace_distrib, vec)

        return self.meshspace_vector.copy()

    def interp_mesh_to_data(self, input_mesh_vector):
        """
        Function to interpolate from the FEM mesh to the sensor locations

        This function takes a distributed mesh vector and interpolates it
        to a gathered numpy array on the root process (all other processes
        will return an empty array). Input must be a Firedrake Vector
        defined on the FEM mesh.

        :param input_mesh_vector: Firedrake Vector to be interpolated to
                                  the mesh locations.
        :type input_mesh_vector: Firedrake Vector
        :returns: Numpy array holding interpolated solution at the
                  sensor location. Root process will hold all data and
                  return a 1D array of length ``(n_data,)``, while all
                  other processes will return an array of zero length.
        :rtype: ndarray
        """

        if not self.is_assembled:
            self.assemble()

        # check vector local sizes and copy values
        if not isinstance(input_mesh_vector, Vector):
            raise TypeError("input_mesh_vector must be a firedrake vector")
        assert (input_mesh_vector.local_size() == self.meshspace_vector.local_size() and
                input_mesh_vector.size() == self.meshspace_vector.size()), "input vector must be the same length as the FEM DOFs and be distributed across processes in the same way"
        self.meshspace_vector.set_local(input_mesh_vector.get_local())

        # interpolate to dataspace and gather, returning numpy array

        with self.meshspace_vector.dat.vec_ro as b:
            self.interp.multTranspose(b, self.dataspace_distrib)

        self._gather()

        return np.copy(self.dataspace_gathered.array)

    def get_meshspace_column_vector(self, idx):
        """
        Returns distributed meshspace column vector for a given data point

        The FEM solve requires extracting the column vectors (distributed
        over the full mesh) for each sensor location. This function
        checks that the index is valid and returns the appropriate
        column vector as a Firedrake Vector.

        :param idx: Index of desired sensor location. Must be a non-negative
                    integer less than the total number of sensors.
        :type idx: int
        :returns: Firedrake Vector holding the appropriate column of
                  the interpolation matrix.
        :rtype: Firedrake Vector
        """

        assert idx >= 0 and idx < self.n_data, "idx out of range"

        if not self.is_assembled:
            self.assemble()

        f = Function(self.function_space).vector()

        f.set_local(self.interp.getColumnVector(idx).array)

        return f

    def __str__(self):
        """
        Returns a string representation of interpolation matrix

        :returns: String representation of the interpolation matrix
        :rtype: str
        """

        return "Interpolation matrix from {} mesh points to {} data points".format(self.n_mesh, self.n_data)

def interpolate_cell(data_coord, nodal_points):
    """
    Interpolate between nodal points and data point

    This function is used to interpolate the desired sensor location
    to the FEM DOFs of the cell in which the sensor point lies. Returns
    an array of floats that determine the appropriate weight for each FEM
    DOF given the location of the sensor.

    **Note:** At present, this routine technically does a projection by
    performing a least squares solution in the case that there are more
    constraints than free parameters, rather than true interpolation. The
    two are identical for linear basis functions on interval, triangular,
    or tetrahedral meshes, but are not guaranteed to match for higher
    order basis functions or other meshes. However, this should always
    return a reasonable approximation. Future improvements to ``stat-fem``
    will fix this by integrating more closely with Firedrake to compute
    the basis function values directly at the sensor location.

    :param data_coord: Spatial location of the sensor. Must be a 1D numpy
                       array of the same length as the spatial dimension
                       of the FEM problem.
    :type data_coord: ndarray
    :param nodal_points: Spatial locations of the FEM DOFs. Must be a 2D
                         numpy array with the first index representing
                         the number of nodal points and the second index
                         representing the spatial dimension (must match
                         the length of ``data_coord``).
    :type nodal_points: ndarray
    :returns: 1D numpy array holding the weights for the different nodal points.
              Will have length corresponding to the number of nodal points
              in the FEM cell. The sum of the array elements will be 1.
    :rtype: ndarray
    """

    if nodal_points.ndim == 1:
        nodal_points = np.reshape(nodal_points, (-1, 1))
    npts, ndim = nodal_points.shape
    assert len(data_coord) == ndim, "data provided for interpolation has the wrong number of spatial dimensions"

    A = np.ones((ndim + 1, npts))
    A[0:-1,:] = np.transpose(nodal_points)
    b = np.array(list(data_coord) + [1.])

    val = np.linalg.lstsq(A, b, rcond=None)[0]

    eps = 1.e-10

    assert np.abs(np.sum(val) - 1.) <= eps, "interpolation between data and coordinates failed"

    return val
