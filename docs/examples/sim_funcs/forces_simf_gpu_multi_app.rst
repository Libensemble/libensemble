Template with dynamic resources (CPU app and GPU app)
-----------------------------------------------------

.. role:: underline
   :class: underline

Launches either the CPU-only or GPU version of the forces MPI app and auto-assigns
ranks and GPU resources as requested by the generator.

This makes efficient use of each node as the expensive GPU simulations will use the GPUs on
the node/s, while the rest of the CPU cores are assigned to the simple CPU-only simulations.

See this publication_ for a real-world demonstration of these capabilities.

.. automodule:: forces_multi_app.forces_simf
   :members:
   :undoc-members:

.. dropdown:: :underline:`forces_simf.py`

   .. literalinclude:: ../../../libensemble/tests/scaling_tests/forces/forces_multi_app/forces_simf.py
      :language: python
      :linenos:

The generator in the example below assigns to each simulation either the CPU application
or the GPU application and also randomly assigns the number of processors for each
simulation. For the GPU application, one GPU is used for each MPI rank. As many nodes
as necessary will be used for each application.

The special generator output fields "num_procs" and "num_gpus" are automatically picked
up by each worker and these will be used when the simulation is run, unless overridden.

.. dropdown:: :underline:`Example usage`

   .. literalinclude:: ../../../libensemble/tests/scaling_tests/forces/forces_multi_app/run_libe_forces.py
      :language: python
      :linenos:

More information is available in the :doc:`Forces GPU tutorial <../../tutorials/forces_gpu_tutorial>`
and the video_ demonstration on Frontier_.

.. _Frontier: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
.. _publication: https://doi.org/10.1103/PhysRevAccelBeams.26.084601
.. _video: https://www.youtube.com/watch?v=H2fmbZ6DnVc
