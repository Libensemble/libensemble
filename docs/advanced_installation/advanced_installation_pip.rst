pip
===

`Advanced Installation <advanced_installation.html>`__ \|\| **pip** \|\| `uv <advanced_installation_uv.html>`__ \|\| `pixi <advanced_installation_pixi.html>`__ \|\| `conda <advanced_installation_conda.html>`__ \|\| `Spack <advanced_installation_spack.html>`__

To install the latest PyPI_ release::

    pip install libensemble

To pip install libEnsemble from the latest develop branch::

    python -m pip install --upgrade git+https://github.com/Libensemble/libensemble.git@develop

**Installing with mpi4py**

If you wish to use ``mpi4py`` with libEnsemble (choosing MPI out of the three
:doc:`communications options<../running_libE>`), then this should
be installed to work with the existing MPI on your system. For example,
the following line::

    pip install mpi4py

will use the ``mpicc`` compiler wrapper on your PATH to identify the MPI library.
To specify a different compiler wrapper, add the ``MPICC`` option.
You also may wish to avoid existing binary builds; for example,::

    MPICC=mpiicc pip install mpi4py --no-binary mpi4py

.. _PyPI: https://pypi.org
