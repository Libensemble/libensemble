Simulation Functions
====================

Below are example simulation functions available in libEnsemble.
Most of these demonstrate an inexpensive algorithm and do not
launch tasks (user applications). To see an example of a simulation
function launching tasks, see the
:doc:`Electrostatic Forces tutorial <../tutorials/executor_forces_tutorial>`.

.. IMPORTANT::
  See the API for simulation functions :ref:`here<api_sim_f>`.

.. role:: underline
    :class: underline

Simple simulation functions
---------------------------
Ideal for simple debugging of generator processes or system testing.

.. toctree::
   :maxdepth: 1

   Borehole function <sim_funcs/borehole>
   Borehole function with kills <sim_funcs/borehole_kills>
   Chwirut1 vector-valued function <sim_funcs/chwirut1>
   Inverse Bayesian likelihood <sim_funcs/inverse_bayes>
   Norm <sim_funcs/simple_sim>
   Rosenbrock test optimization function <sim_funcs/rosenbrock>
   Six Hump Camel <sim_funcs/six_hump_camel>
   Test noisy function <sim_funcs/noisy_vector_mapping>
   Test periodic function <sim_funcs/periodic_func>

Functions that run user applications
------------------------------------

These use the executor to launch applications and in some cases
handle dynamic CPU/GPU allocation.

The ``Variable resources`` module contains basic examples, while the ``Template``
examples use a simple MPI/OpenMP (with GPU offload option) application (``forces``)
to demonstrate libEnsemble’s capabilities on various HPC systems. The
build_forces.sh_ file gives compile lines for building the simple ``forces``
application on various platforms (use -DGPU to build for GPU).

.. toctree::
   :maxdepth: 1

   Variable resources <sim_funcs/var_resources>
   sim_funcs/forces_simf
   sim_funcs/forces_simf_input_file
   sim_funcs/forces_simf_gpu
   sim_funcs/forces_simf_gpu_vary_resources
   sim_funcs/forces_simf_gpu_multi_app
   WarpX Example<https://libensemble.readthedocs.io/projects/libe-community-examples/en/latest/simulators.html#module-warpx-simf>

Special simulation functions
----------------------------

.. toctree::
   :maxdepth: 1

   sim_funcs/mock_sim

.. _build_forces.sh: https://github.com/Libensemble/libensemble/blob/main/libensemble/tests/scaling_tests/forces/forces_app/build_forces.sh
