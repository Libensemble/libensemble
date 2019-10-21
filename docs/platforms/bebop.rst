=====
Bebop
=====

Bebop_ is the newest addition to the computational power of LCRC at Argonne
National Laboratory, featuring both Intel Broadwell and Knights Landing nodes.



Before getting started
----------------------

An Argonne LCRC_ account is required to access Bebop. Interested users will need
to apply for and be granted an account before continuing.



Configuring Python
------------------

Begin by loading the Python 3 Anaconda_ module::

    module load anaconda3/2018.12

Create a Conda_ virtual environment in which to install libEnsemble and all
dependencies::

    conda config --add channels intel
    conda create --name my_env intelpython3_core python=3
    source activate my_env

You should have an indication that the virtual environment is activated.
Install mpi4py_ and libEnsemble in this environment, making sure to reference
the pre-installed Intel MPI Compiler. Your prompt should be similar to the
following block:

.. code-block:: console

    (my_env) user@beboplogin4:~$ CC=mpiicc MPICC=mpiicc pip install mpi4py --no-binary mpi4py
    (my_env) user@beboplogin4:~$ pip install libensemble


.. _Bebop: https://www.lcrc.anl.gov/systems/resources/bebop/
.. _LCRC: https://www.lcrc.anl.gov
.. _Anaconda: https://www.anaconda.com/distribution/
.. _Conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
