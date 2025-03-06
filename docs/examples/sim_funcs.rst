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

.. toctree::
   :maxdepth: 1

   sim_funcs/simple_sim
   sim_funcs/six_hump_camel
   sim_funcs/borehole
   sim_funcs/borehole_kills
   sim_funcs/chwirut1
   sim_funcs/inverse_bayes
   sim_funcs/noisy_vector_mapping
   sim_funcs/periodic_func
   sim_funcs/rosenbrock
   sim_funcs/surmise_test_function


Functions that run user applications
------------------------------------

These use the executor to launch applications and in some cases
handle dynamic CPU/GPU allocation.

The ``var_resources`` module contains basic examples, while the ``forces``
examples use an MPI/OpenMP (with GPU offload option) application that is used
to demonstrate libEnsemble’s capabilities on various HPC systems. The
build_forces.sh_ file gives compile lines for building the forces application
on various platforms (use -DGPU to build for GPU).

.. toctree::
   :maxdepth: 1

   sim_funcs/var_resources
   sim_funcs/forces_simf
   sim_funcs/forces_simf_input_file
   sim_funcs/forces_simf_gpu
   sim_funcs/forces_simf_gpu_vary_resources
   sim_funcs/forces_simf_gpu_multi_app


Special simulation functions
----------------------------

.. toctree::
   :maxdepth: 1

   sim_funcs/mock_sim

.. _build_forces.sh: https://github.com/Libensemble/libensemble/blob/main/libensemble/tests/scaling_tests/forces/forces_app/build_forces.sh
