========
Frontier
========

Frontier_ is an HPE Cray EX exascale system located at Oak Ridge
Leadership Computing Facility (OLCF).

Each Frontier compute node contains one 64-core AMD EPYC and four AMD MI250X GPUs
(eight logical GPUs).

Frontier uses the SLURM scheduler to submit jobs from login nodes to run on the
compute nodes.

Installing libEnsemble
----------------------

Begin by loading the ``python`` module::

    module load cray-python

You may wish to create a virtual environment to install packages in (see python_on_frontier_).

.. dropdown:: Example of using virtual environment

   Having created a dir ``/ccs/proj/<project_id>/libensemble``:

   .. code-block:: console

       python -m venv /ccs/proj/<project_id>/libensemble/libe_env
       source /ccs/proj/<project_id>/libensemble/libe_env/bin/activate

libEnsemble can be installed via pip::

    pip install libensemble

See :doc:`advanced installation<../advanced_installation>` for other installation options.

Example
-------

Note that a video demonstration_ of this example is also available.

To run the :doc:`forces_gpu<../tutorials/forces_gpu_tutorial>` tutorial on Frontier.

To obtain the example you can git clone libEnsemble - although only
the forces sub-directory is needed::

    git clone https://github.com/Libensemble/libensemble
    cd libensemble/libensemble/tests/scaling_tests/forces/forces_app

To compile forces::

    module load rocm
    module load craype-accel-amd-gfx90a
    cc -DGPU -I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64 -fopenmp -O3 -o forces.x forces.c

Now go to forces_gpu directory::

    cd ../forces_gpu

Now grab an interactive session on one node::

    salloc --nodes=1 -A <project_id> --time=00:10:00

Then in the session run::

    python run_libe_forces.py --nworkers 9

This places the generator on the first worker and runs simulations on the
others (each simulation using one GPU).

To see GPU usage, ssh into the node you are on in another window and run::

    module load rocm
    watch -n 0.1 rocm-smi

.. _Frontier: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
.. _python_on_frontier: https://www.olcf.ornl.gov/wp-content/uploads/2-16-23_python_on_frontier.pdf
.. _demonstration: https://youtu.be/H2fmbZ6DnVc
