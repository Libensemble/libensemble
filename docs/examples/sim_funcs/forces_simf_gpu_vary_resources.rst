Template for GPU executables with dynamic resources
---------------------------------------------------

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

   Note the use of the generator function ``uniform_sample_with_var_gpus``  that sets ``num_gpus`` as a ``gen_specs`` output field corresponding to each generated simulation input.

   The special generator output field "num_gpus" is automatically picked up by each worker
   and will be used when the simulation is run, unless overridden.

   .. literalinclude:: ../../../libensemble/tests/scaling_tests/forces/forces_gpu_var_resources/run_libe_forces.py
      :language: python
      :linenos:

Also see the :doc:`Forces GPU tutorial <../../tutorials/forces_gpu_tutorial>` and the
video_ demonstration on Frontier_.

.. _video: https://www.youtube.com/watch?v=H2fmbZ6DnVc
.. _Frontier: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
