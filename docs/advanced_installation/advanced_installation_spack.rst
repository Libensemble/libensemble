Spack
=====

`Advanced Installation <advanced_installation.html>`__ \|\| `pip <advanced_installation_pip.html>`__ \|\| `uv <advanced_installation_uv.html>`__ \|\| `pixi <advanced_installation_pixi.html>`__ \|\| `conda <advanced_installation_conda.html>`__ \|\| **Spack**

Install libEnsemble using the Spack_ distribution::

    spack install py-libensemble

The above command will install the latest release of libEnsemble with
the required dependencies only. Other optional
dependencies can be specified through variants. The following
line installs libEnsemble version 1.5.0 with some common variants
(e.g., using :doc:`APOSMM<../examples/gest_api/aposmm>`):

.. code-block:: bash

    spack install py-libensemble @1.5.0 +mpi +scipy +mpmath +petsc4py +nlopt

The list of variants can be found by running::

    spack info py-libensemble

On some platforms you may wish to run libEnsemble without ``mpi4py``,
using a serial PETSc build. This is often preferable if running on
the launch nodes of a three-tier system::

    spack install py-libensemble +scipy +mpmath +petsc4py ^py-petsc4py~mpi ^petsc~mpi~hdf5~hypre~superlu-dist

The installation will create modules for libEnsemble and the dependent
packages. These can be loaded by running::

    spack load -r py-libensemble

Any Python packages will be added to the PYTHONPATH when the modules are loaded. If you do not have
modules on your system you may need to install ``lmod`` (also available in Spack)::

    spack install lmod
    . $(spack location -i lmod)/lmod/lmod/init/bash
    spack load lmod

Alternatively, Spack could be used to build the serial ``petsc4py``, and Conda could use this by loading
the ``py-petsc4py`` module thus created.

**Hint**: When combining Spack and Conda, you can access your Conda Python and packages in your
``~/.spack/packages.yaml`` while your Conda environment is activated, using ``CONDA_PREFIX``
For example, if you have an activated Conda environment with Python 3.11 and SciPy installed:

.. code-block:: yaml

    packages:
    python:
        externals:
        - spec: "python"
        prefix: $CONDA_PREFIX
        buildable: False
    py-numpy:
        externals:
        - spec: "py-numpy"
        prefix: $CONDA_PREFIX/lib/python3.11/site-packages/numpy
        buildable: False
    py-scipy:
        externals:
        - spec: "py-scipy"
        prefix: $CONDA_PREFIX/lib/python3.11/site-packages/scipy
        buildable: True

For more information on Spack builds and any particular considerations
for specific systems, see the spack_libe_ repository. In particular, this
includes some example ``packages.yaml`` files (which go in ``~/.spack/``).
These files are used to specify dependencies that Spack must obtain from
the given system (rather than building from scratch). This may include
``Python`` and the packages distributed with it (e.g., ``numpy``), and will
often include the system MPI library.

.. _Spack: https://spack.readthedocs.io/en/latest
.. _spack_libe: https://github.com/Libensemble/spack_libe
