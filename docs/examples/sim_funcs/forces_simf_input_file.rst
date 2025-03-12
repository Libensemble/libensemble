Template for CPU executables with input file
--------------------------------------------

.. role:: underline
   :class: underline

Many applications read an input file instead of being given parameters directly
on the run line.

In this variant of the forces example, a templated input file is parameterized
for each evaluation.

This requires **jinja2** to be installed::

    pip install jinja2

In the example, the file ``forces_input`` contains the following (remember
we are using particles as seed also for simplicity)::

    num_particles = {{particles}}
    num_steps = 10
    rand_seed = {{particles}}

libEnsemble will copy this input file to each simulation directory. There, the
simulation function will updates the input file with the ``particles`` value for
this simulation.

.. automodule:: forces_simple_with_input_file.forces_simf
   :members: run_forces

.. dropdown:: :underline:`forces_simf.py`

   .. literalinclude:: ../../../libensemble/tests/scaling_tests/forces/forces_simple_with_input_file/forces_simf.py
      :language: python
      :linenos:

.. dropdown:: :underline:`Example usage`

   .. literalinclude:: ../../../libensemble/tests/scaling_tests/forces/forces_simple_with_input_file/run_libe_forces.py
      :language: python
      :linenos:

Also see the :doc:`Forces tutorial <../../tutorials/executor_forces_tutorial>`.
