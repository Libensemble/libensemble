forces (GPU) dynamic resources
------------------------------

.. role:: underline
   :class: underline

.. automodule:: forces_gpu_var_resources.forces_simf
   :members:
   :undoc-members:

.. dropdown:: :underline:`forces_simf.py`
 
   .. literalinclude:: ../../../libensemble/tests/scaling_tests/forces/forces_gpu_var_resources/forces_simf.py
      :language: python
      :linenos:


.. dropdown:: :underline:`Example usage`

   Note the use of the generator function ``uniform_sample_with_var_gpus``  that sets ``num_gpus`` As a gen specs out for each generated input set (for the simulation).

   .. literalinclude:: ../../../libensemble/tests/scaling_tests/forces/forces_gpu_var_resources/run_libe_forces.py
      :language: python
      :linenos:
