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

six_hump_camel
--------------
.. automodule:: six_hump_camel
  :members:
  :undoc-members:

.. container:: toggle

   .. container:: header

      :underline:`six_hump_camel.py`

   .. literalinclude:: ../../libensemble/sim_funcs/six_hump_camel.py
      :language: python
      :linenos:

chwirut
-------
.. automodule:: chwirut1
  :members:
  :undoc-members:

noisy_vector_mapping
--------------------
.. automodule:: noisy_vector_mapping
  :members:
  :undoc-members:

.. container:: toggle

   .. container:: header

      :underline:`noisy_vector_mapping.py`

   .. literalinclude:: ../../libensemble/sim_funcs/noisy_vector_mapping.py
      :language: python
      :linenos:

periodic_func
-------------
.. automodule:: periodic_func
  :members:
  :undoc-members:

borehole
--------
.. automodule:: borehole
  :members:
  :undoc-members:

executor_hworld
---------------
.. automodule:: executor_hworld
  :members:

.. container:: toggle

   .. container:: header

      :underline:`executor_hworld.py`

   .. literalinclude:: ../../libensemble/sim_funcs/executor_hworld.py
      :language: python
      :linenos:
