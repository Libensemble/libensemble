Advanced Installation
=====================

libEnsemble can be installed from ``pip``, ``Conda``, or ``Spack``.

libEnsemble requires the following dependencies, which are typically
automatically installed alongside libEnsemble:

* Python_       ``>= 3.10``
* NumPy_        ``>= 1.21``
* psutil_       ``>= 5.9.4``
* `pydantic`_   ``>= 1.10.12``
* pyyaml_       ``>= v6.0``
* tomli_        ``>= 1.2.1``

Given libEnsemble's compiled dependencies, the following installation
methods each offer a trade-off between convenience and the ability
to customize builds, including platform-specific optimizations.

We always recommend installing in a virtual environment from Conda or another source.

Further recommendations for selected HPC systems are given in the
:ref:`HPC platform guides<platform-index>`.

.. tab-set::

    .. tab-item:: pip

        To install the latest PyPI_ release::

            pip install libensemble

        To pip install libEnsemble from the latest develop branch::

            python -m pip install --upgrade git+https://github.com/Libensemble/libensemble.git@develop

        **Installing with mpi4py**

        If you wish to use ``mpi4py`` with libEnsemble (choosing MPI out of the three
        :doc:`communications options<running_libE>`), then this should
        be installed to work with the existing MPI on your system. For example,
        the following line::

            pip install mpi4py

        will use the ``mpicc`` compiler wrapper on your PATH to identify the MPI library.
        To specify a different compiler wrapper, add the ``MPICC`` option.
        You also may wish to avoid existing binary builds; for example,::

            MPICC=mpiicc pip install mpi4py --no-binary mpi4py

        On Summit, the following line is recommended (with gcc compilers)::

            CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

    .. tab-item:: conda

        Install libEnsemble with Conda_ from the conda-forge channel::

            conda config --add channels conda-forge
            conda install -c conda-forge libensemble

        This package comes with some useful optional dependencies, including
        optimizers and will install quickly as ready binary packages.

        **Installing with mpi4py with Conda**

        If you wish to use ``mpi4py`` with libEnsemble (choosing MPI out of the three
        :doc:`communications options<running_libE>`), you can use the
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

    .. tab-item:: Spack

        Install libEnsemble using the Spack_ distribution::

            spack install py-libensemble

        The above command will install the latest release of libEnsemble with
        the required dependencies only. Other optional
        dependencies can be specified through variants. The following
        line installs libEnsemble version 0.7.2 with some common variants
        (e.g., using :doc:`APOSMM<../examples/aposmm>`):

        .. code-block:: bash

            spack install py-libensemble @0.7.2 +mpi +scipy +mpmath +petsc4py +nlopt

        The list of variants can be found by running::

            spack info py-libensemble

        On some platforms you may wish to run libEnsemble without ``mpi4py``,
        using a serial PETSc build. This is often preferable if running on
        the launch nodes of a three-tier system (e.g., Summit)::

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
        For example, if you have an activated Conda environment with Python 3.10 and SciPy installed:

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
                prefix: $CONDA_PREFIX/lib/python3.10/site-packages/numpy
                buildable: False
            py-scipy:
                externals:
                - spec: "py-scipy"
                prefix: $CONDA_PREFIX/lib/python3.10/site-packages/scipy
                buildable: True

        For more information on Spack builds and any particular considerations
        for specific systems, see the spack_libe_ repository. In particular, this
        includes some example ``packages.yaml`` files (which go in ``~/.spack/``).
        These files are used to specify dependencies that Spack must obtain from
        the given system (rather than building from scratch). This may include
        ``Python`` and the packages distributed with it (e.g., ``numpy``), and will
        often include the system MPI library.

Optional Dependencies for Additional Features
---------------------------------------------

The following packages may be installed separately to enable additional features:

* pyyaml_ and tomli_ - Parameterize libEnsemble via yaml or toml
* `Globus Compute`_ - Submit simulation or generator function instances to remote Globus Compute endpoints

.. _conda-forge: https://conda-forge.org/
.. _Conda: https://docs.conda.io/en/latest/
.. _GitHub: https://github.com/Libensemble/libensemble
.. _Globus Compute: https://www.globus.org/compute
.. _MPICH: https://www.mpich.org/
.. _NumPy: http://www.numpy.org
.. _Open MPI: https://www.open-mpi.org/
.. _psutil: https://pypi.org/project/psutil/
.. _pydantic: https://docs.pydantic.dev/1.10/
.. _PyPI: https://pypi.org
.. _Python: http://www.python.org
.. _pyyaml: https://pyyaml.org/
.. _Spack: https://spack.readthedocs.io/en/latest
.. _spack_libe: https://github.com/Libensemble/spack_libe
.. _tomli: https://pypi.org/project/tomli/
.. _tqdm: https://tqdm.github.io/
