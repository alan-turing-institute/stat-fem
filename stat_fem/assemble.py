from .ForcingCovariance import ForcingCovariance
from .InterpolationMatrix import InterpolationMatrix

def assemble(f):
    """
    Assembly function provided for assembling ``stat-fem`` PETSc matrices
    
    Functional interface to assembling PETSc matrices needed for the ``stat-fem``
    solves. Note that this is just for convenience and simply calls the ``assemble``
    method of the provided object. Note that if the matrices are not assembled
    when they are required, ``stat-fem`` will assemble them automatically.

    :param f: a :class:`~stat_fem.ForcingCovariance` or :class:`~stat_fem.InterpolationMatrix`
    :type f: ForcingCovariance or InterpolationMatrix
    :returns: Assembled object of the appropriate type.
    """

    if isinstance(f, (ForcingCovariance, InterpolationMatrix)):
        f.assemble()
        return f
    else:
        raise NotImplementedError("The stat-fem assemble function no longer supports assembling " +
                                  "Firedrake objects. Use the firedrake assemble function directly")


