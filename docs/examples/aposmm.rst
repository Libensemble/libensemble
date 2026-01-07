APOSMM
------

Asynchronously Parallel Optimization Solver for finding Multiple Minima
(APOSMM) coordinates concurrent local optimization runs to identify
many local minima faster on parallel hardware.

Supported local optimization routines include:

  - DFO-LS_ Derivative-free solver for (bound constrained) nonlinear least-squares minimization
  - NLopt_ Library for nonlinear optimization, providing a common interface for various methods
  - `scipy.optimize`_ Open-source solvers for nonlinear problems, linear programming,
    constrained and nonlinear least-squares, root finding, and curve fitting.
  - `PETSc/TAO`_ Routines for the scalable (parallel) solution of scientific applications

Required: mpmath_, SciPy_

Optional (see below): petsc4py_, nlopt_, DFO-LS_

Configuring APOSMM
^^^^^^^^^^^^^^^^^^

APOSMM works with a choice of optimizers, some requiring external packages. Specify
them on a global level before importing APOSMM::

    import libensemble.gen_funcs
    libensemble.gen_funcs.rc.aposmm_optimizers = <optimizers>

where ``optimizers`` is a string (or list-of-strings) from:

``"petsc"``, ``"nlopt"``, ``"dfols"``, ``"scipy"``, ``"external"``

.. dropdown:: Issues with ensemble hanging or failed simulations with PETSc?

    If using the MPIExecutor or other MPI routines
    and your MPI backend is Open MPI, then you must:

    - Use ``local`` comms for libEnsemble (no ``mpirun``, ``mpiexec``, ``aprun``, etc.).
    - Must **NOT** include the *aposmm_optimizers* line above.

    This is because PETSc imports MPI, and a global import of PETSc results
    in nested MPI (which is not supported by Open MPI).

To see the optimization algorithms supported, see `LocalOptInterfacer`_.

.. seealso::

    :doc:`Persistent APOSMM Tutorial<../tutorials/aposmm_tutorial>`

Persistent APOSMM
^^^^^^^^^^^^^^^^^

.. automodule:: persistent_aposmm
  :members: aposmm
  :undoc-members:

LocalOptInterfacer
^^^^^^^^^^^^^^^^^^
.. automodule:: aposmm_localopt_support
  :members: LocalOptInterfacer
  :undoc-members:

.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _mpmath: https://pypi.org/project/mpmath
.. _nlopt: https://nlopt.readthedocs.io/en/latest/
.. _petsc4py: https://bitbucket.org/petsc/petsc4py
.. _SciPy: https://pypi.org/project/scipy
.. _PETSc/TAO: http://www.mcs.anl.gov/petsc
.. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html
