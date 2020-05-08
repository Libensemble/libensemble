Advanced Installation
=====================

.. note::
    At the current time we recommend new users start with the develop branch
    as this has many additions, including API breaking changes since v0.6.0.
    This applies until the release of v0.7.0

libEnsemble can be installed from ``pip``, ``Conda``, or ``Spack``.

In view of libEnsemble's compiled dependencies, these approaches
offer a trade-off between convenience and the ability
to customize builds, including platform specific optimizations.

Further recommendations for selected HPC systems is given in the
:doc:`HPC platform guide<platforms/platforms_index>`.

pip
---

We always recommend installing in a virtual environment such as Conda.
If not, then use ``pip3``, ``python3`` below.

To install the latest pip release::

    pip install libensemble

The above comes with required dependencies only. To install with some
common user dependencies (as used in the examples/tests)::

    pip install libensemble[extras]

Note that ``PETSc`` will build from source so may take a while.

To pip install libEnsemble from the latest develop branch::

    python -m pip install --upgrade git+https://github.com/Libensemble/libensemble.git@develop


Installing with mpi4py
^^^^^^^^^^^^^^^^^^^^^^

If you wish to use ``mpi4py`` with libEnsemble (out of the three
:doc:`commuications options<running_libE>`), then this should
be installed to work with the existing MPI on your system. For example,
the following line::

    pip install mpi4py

will use the ``mpicc`` compiler wrapper on your PATH to identify the MPI.
To specify a different compiler wrapper, add the ``MPICC`` option.
You also may wish to avoid existing binary builds e.g.::

    MPICC=mpiicc pip install mpi4py --no-binary mpi4py

On Summit, the following line is recommended (with gcc compilers)::

    CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py


Conda
-----

Install libEnsemble with Conda_ from the conda-forge channel::

    conda config --add channels conda-forge
    conda install -c conda-forge libensemble

This package comes with some useful optional dependencies, including
optimizers and will install quickly as ready binary packages.


Installing with mpi4py with Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to use ``mpi4py`` with libEnsemble (out of the three
:doc:`commuications options<running_libE>`) you can use the
following.

.. note::
    For clusters and HPC systems, always install ``mpi4py`` to use the
    system MPI library (see pip instructions above).

For a standalone build that comes with an MPI implementation, you can install
libEnsemble using one of the following variants.

To install libEnsemble with MPICH_::

    conda install -c conda-forge libensemble=*=mpi_mpich*

To install libEnsemble with `Open MPI`_::

    conda install -c conda-forge libensemble=*=mpi_openmpi*

The asterisks will pick up the latest version and build.

.. note::
    This syntax may not work without adjustments on macOS or any non-bash
    shell. Try this instead, for example::

        conda install -c conda-forge libensemble='*'=mpi_mpich'*'

For a complete list of builds for libEnsemble on Conda::

    conda search libensemble --channel conda-forge


Spack
-----

Install libEnsemble using the Spack_ distribution::

    spack install py-libensemble

The above command will install the required dependencies only. There
are several other optional dependencies that can be specified
through variants. The following line installs libEnesmble
version 0.7.0 with all the variants::

    spack install py-libensemble @0.7.0 +mpi +scipy +petsc4py +nlopt

On some platforms you may wish to run libEnsemble without mpi4py,
using a serial PETSc build (this is often preferable if running on
the launch nodes of a three-tier system (e.g. Theta/Summit)::

    spack install py-libensemble @0.7.0 +scipy +petsc4py~mpi

The install will create modules for libEnsemble and the dependent
packages. These can be loaded by::

    spack load -r py-libensemble

For more information on Spack builds and any particular considerations
for specific systems, see the spack_libe_ repostory. In particular, this
includes some example ``packages.yaml`` files (which go in ``~/.spack/``).
These files are used to specify dependencies that Spack must obtain from
the given system (rather than building from scratch). This may include
``Python`` and the packages disributed with it (e.g. ``numpy``), and will
often include the system MPI library.


.. _GitHub: https://github.com/Libensemble/libensemble
.. _Conda: https://docs.conda.io/en/latest/
.. _conda-forge: https://conda-forge.org/
.. _MPICH: https://www.mpich.org/
.. _`Open MPI`: https://www.open-mpi.org/
.. _Spack: https://spack.readthedocs.io/en/latest
.. _spack_libe: https://github.com/Libensemble/spack_libe

