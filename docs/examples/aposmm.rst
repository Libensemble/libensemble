APOSMM
------

Asynchronously Parallel Optimization Solver for finding Multiple Minima
(APOSMM) coordinates concurrent local optimization runs in order to identify
many local minima.

Required: mpmath_, SciPy_

Optional (see below): petsc4py_, nlopt_, DFO-LS_

Configuring APOSMM
^^^^^^^^^^^^^^^^^^

APOSMM works with a choice of optimizers, some requiring external packages. To
import the optimization packages (and their dependencies) at a global level
(recommended), add the following lines in the calling script before importing
APOSMM::

    import libensemble.gen_funcs
    libensemble.gen_funcs.rc.aposmm_optimizers = <optimizers>

where ``optimizers`` is a string (or list of strings) from the available options:

``"petsc"``, ``"nlopt"``, ``"dfols"``, ``"scipy"``, ``"external"``

.. dropdown:: Issues with ensemble hanging or failed simulations?

    Note that if using **mpi4py** comms, PETSc must be imported at the global
    level or the ensemble may hang.

    Exception: In the case that you are using the MPIExecutor or other MPI inside
    a user function and you are using Open MPI, then you must:

    - Use ``local`` comms for libEnsemble (not ``mpi4py``)
    - Must **NOT** include the *rc* line above

    This is because PETSc imports MPI, and a global import of PETSc would result
    in nested MPI (which is not supported by Open MPI). When the above line is
    not used, an import local to the optimization function will happen.

To see the optimization algorithms supported, see `LocalOptInterfacer`_.

.. seealso::

    :doc:`Persistent APOSMM Tutorial<../tutorials/aposmm_tutorial>`

Persistent APOSMM
^^^^^^^^^^^^^^^^^

.. automodule:: persistent_aposmm
  :members:
  :undoc-members:

LocalOptInterfacer
^^^^^^^^^^^^^^^^^^
.. automodule:: aposmm_localopt_support
  :members:
  :undoc-members:

.. _DFO-LS: https://github.com/numericalalgorithmsgroup/dfols
.. _mpmath: https://pypi.org/project/mpmath
.. _nlopt: https://nlopt.readthedocs.io/en/latest/
.. _petsc4py: https://bitbucket.org/petsc/petsc4py
.. _SciPy: https://pypi.org/project/scipy
