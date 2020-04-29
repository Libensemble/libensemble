Advanced Installation
=====================

Installing the Develop Version or Other Branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can check out development versions or branches of libEnsemble like
``develop`` by cloning the GitHub_ repository, installing it with ``pip``, then
choosing a branch with ``git``. Keep in mind branches installed this way aren't
as vigorously tested as the official releases.

Clone the libEnsemble repository from GitHub::

    git clone https://github.com/Libensemble/libensemble.git
    cd libensemble

Install this repository::

    pip install -e .

By default, Python should have access to the code on libEnsemble's
``master`` branch. Switch to the ``develop`` branch::

    git checkout develop

Variants from conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~

libEnsemble can be installed from the Conda_ conda-forge_ channel alongside
specific versions of MPI.

To install libEnsemble with MPICH_::

    conda install -c conda-forge libensemble=*=mpi_mpich*

To install libEnsemble with `Open MPI`_::

    conda install -c conda-forge libensemble=*=mpi_openmpi*

.. note::
    This syntax may not work without adjustments on macOS or any non-bash
    shell environment. Try this instead, for example::

        conda install -c conda-forge libensemble='*'=mpi_mpich'*'

For a complete list of builds for libEnsemble on Conda::

    conda search libensemble --channel conda-forge

Variants from Spack
~~~~~~~~~~~~~~~~~~~

[Information goes here]


.. _GitHub: https://github.com/Libensemble/libensemble
.. _Conda: https://docs.conda.io/en/latest/
.. _conda-forge: https://conda-forge.org/
.. _MPICH: https://www.mpich.org/
.. _`Open MPI`: https://www.open-mpi.org/
