======
Improv
======

Improv_ is an LCRC production cluster at Argonne National Laboratory with 825
dual-socket compute nodes with AMD 7713 64-core processors.

Installing libEnsemble and Dependencies
---------------------------------------

To create a conda environment and install libEnsemble::

    module load anaconda3
    conda create --name improv_libe_env python=3
    conda activate improv_libe_env
    pip install libensemble

See :doc:`here<../advanced_installation>` for more information on advanced
options for installing libEnsemble, including using Spack.

Job Submission
--------------

Improv uses the PBS scheduler to submit jobs from login nodes to run on compute
nodes.

Example
-------

To run the :doc:`forces_simple<../tutorials/executor_forces_tutorial>` tutorial on Improv.

To obtain the example you can git clone libEnsemble - although only
the forces sub-directory is needed::

    git clone https://github.com/Libensemble/libensemble
    cd libensemble/libensemble/tests/scaling_tests/forces/forces_app

To compile forces::

    module load gcc openmpi aocl
    mpicc -O3 -o forces.x forces.c -lm

Now go to forces_simple directory::

    cd ../forces_simple/

Now grab an interactive session on one node::

    qsub -A <project_id> -l select=1:mpiprocs=64 -l walltime=20:00 -qdebug -I

Once in the interactive session, you may need to reload the modules::

    cd $PBS_O_WORKDIR
    module load anaconda3 gcc openmpi aocl
    conda activate improv_libe_env

Now run forces with five workers (one for generator and four for simulations)::

    python run_libe_forces.py --nworkers 5

mpi4py comms
============

You can install mpi4py as usual having installed the Open-MPI module::

    pip install mpi4py

Note if using ``mpi4py`` comms with Open-MPI, you may need to set ``export OMPI_MCA_coll_hcoll_enable=0``
to prevent HCOLL warnings.

.. _Improv: https://docs.lcrc.anl.gov/improv/running-jobs-improv/
