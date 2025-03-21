=======
Polaris
=======

Polaris_ is a 560-node HPE system located in the ALCF_ at Argonne
National Laboratory. The compute nodes are equipped with one AMD EPYC Milan
processor and four A100 NVIDIA GPUs. It uses the PBS scheduler to submit
jobs from login nodes to run on the compute nodes.

Configuring Python and Installation
-----------------------------------

Python and libEnsemble are available on Polaris with the `conda` module. Load the
``conda`` module and activate the base environment::

    module use /soft/modulefiles
    module load conda; conda activate

This also gives you access to machine-optimized packages such as mpi4py_.

To install further packages, including updating libEnsemble, you may either create
a virtual environment on top of this (if just using ``pip install``) or clone the base
environment (if you need ``conda install``). More details at `Python for Polaris`_.

.. dropdown:: Example of Conda + virtual environment

   To create a virtual environment that allows installation of further packages::

       python -m venv /path/to-venv --system-site-packages
       . /path/to-venv/bin/activate

   where ``/path/to-venv`` can be anywhere you have write access. For future sessions,
   just load the ``conda`` module and run the activate line.

   You can now pip install libEnsemble::

       pip install libensemble

See :doc:`here<../advanced_installation>` for more information on advanced options
for installing libEnsemble, including using Spack.

Job Submission
--------------

Polaris uses the PBS scheduler to submit jobs from login nodes to run on
the compute nodes. libEnsemble runs on the compute nodes using either
``multi-processing`` or ``mpi4py``

A simple example batch script for a libEnsemble use case that runs 5 workers
(e.g., one persistent generator and four for simulations) on one node:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #PBS -A <myproject>
    #PBS -lwalltime=00:15:00
    #PBS -lselect=1
    #PBS -q debug
    #PBS -lsystem=polaris
    #PBS -lfilesystems=home:grand

    export MPICH_GPU_SUPPORT_ENABLED=1

    cd $PBS_O_WORKDIR

    python run_libe_forces.py --nworkers 5

The script can be run with::

    qsub submit_libe.sh

Or you can run an interactive session with::

    qsub -A <myproject> -l select=1 -l walltime=15:00 -lfilesystems=home:grand -qdebug -I

You may need to reload your ``conda`` module and reactivate ``venv`` environment
again after starting the interactive session.

Demonstration
-------------

For an example that runs a small ensemble using a C application (offloading work to the
GPU), see the :doc:`forces_gpu<../tutorials/forces_gpu_tutorial>` tutorial. A video demonstration_
of this example is also available.

.. _ALCF: https://www.alcf.anl.gov/
.. _conda: https://conda.io/en/latest/
.. _demonstration: https://youtu.be/Ff0dYYLQzoU
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Polaris: https://www.alcf.anl.gov/polaris
.. _Python for Polaris: https://docs.alcf.anl.gov/polaris/data-science-workflows/python/
