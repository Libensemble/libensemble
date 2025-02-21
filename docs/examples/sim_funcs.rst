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
   sim_funcs/borehole
   sim_funcs/borehole_kills
   sim_funcs/chwirut1
   sim_funcs/inverse_bayes
   sim_funcs/noisy_vector_mapping
   sim_funcs/periodic_func
   sim_funcs/rosenbrock
   sim_funcs/six_hump_camel
   sim_funcs/surmise_test_function


Functions with CPU/GPU allocation
---------------------------------

.. toctree::
   :maxdepth: 1

   sim_funcs/var_resources
   sim_funcs/forces_simf

Special simulation functions
----------------------------

.. toctree::
   :maxdepth: 1

   sim_funcs/mock_sim


