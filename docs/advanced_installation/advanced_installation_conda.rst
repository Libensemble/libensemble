conda
=====

`Advanced Installation <advanced_installation.html>`__ \|\| `pip <advanced_installation_pip.html>`__ \|\| `uv <advanced_installation_uv.html>`__ \|\| `pixi <advanced_installation_pixi.html>`__ \|\| **conda** \|\| `Spack <advanced_installation_spack.html>`__

Install libEnsemble with Conda_ from the conda-forge channel::

    conda config --add channels conda-forge
    conda install -c conda-forge libensemble

This package comes with some useful optional dependencies, including
optimizers and will install quickly as ready binary packages.

**Installing with mpi4py with Conda**

If you wish to use ``mpi4py`` with libEnsemble (choosing MPI out of the three
:doc:`communications options<../running_libE>`), you can use the
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
    shell. In these cases, try::

        conda install -c conda-forge libensemble='*'=mpi_mpich'*'

For a complete list of builds for libEnsemble on Conda::

    conda search libensemble --channel conda-forge

.. _Conda: https://docs.conda.io/en/latest/
.. _MPICH: https://www.mpich.org/
.. _Open MPI: https://www.open-mpi.org/
