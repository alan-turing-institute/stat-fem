import numpy as np
from firedrake import Function
from firedrake.petsc import PETSc
from firedrake.functionspaceimpl import WithGeometry

class InterpolationMatrix(object):
    "class representing an interpolation matrix"
    def __init__(self, coords, function_space):
        "create and assemble interpolation matrix"

        if not isinstance(function_space, WithGeometry):
            raise TypeError("bad input type for function_space: must be a FunctionSpace")

        self.n_data = coords.shape[0]
        assert (coords.shape[1] == function_space.mesh().cell_dimension()
                ), "shape of coordinates does not match mesh dimension"
        self.n_mesh = function_space.mesh().num_vertices()

        nnz = len(function_space.cell_node_list[0])

        self.interp = PETSc.Mat().createAIJ((self.n_data, self.n_mesh), nnz = nnz)
        self.interp.setUp()

        istart, iend = self.interp.getOwnershipRange()

        for i in range(istart, iend):
            cell = function_space.mesh().locate_cell(coords[i])
            nodes = function_space.cell_node_list[cell]
            for j in nodes:
                temp_array = np.zeros(self.n_mesh)
                temp_array[j] = 1.
                dummy_function = Function(function_space, val = temp_array)
                value = dummy_function.at(coords[i])
                if value > 0.:
                    self.interp.setValue(i, j, value)

        self.interp.assemble()
