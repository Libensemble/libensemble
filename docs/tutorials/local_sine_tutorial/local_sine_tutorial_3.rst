3. Simulator
============

`Introduction <local_sine_tutorial.html>`__ \|\| `1. Getting started <local_sine_tutorial_1.html>`__ \|\| `2. Generator <local_sine_tutorial_2.html>`__ \|\| **3. Simulator** \|\| `4. Script <local_sine_tutorial_4.html>`__ \|\| `5. Next steps <local_sine_tutorial_5.html>`__

Next, we'll write our simulator function or :ref:`sim_f<funcguides-sim>`. Simulator
functions perform calculations based on values from the generator.
:ref:`sim_specs<datastruct-sim-specs>` is a dictionary containing user-defined fields
and parameters.

Create a new Python file named ``sine_sim.py``. Write the following:

.. literalinclude:: ../../../libensemble/tests/functionality_tests/sine_sim.py
    :language: python
    :linenos:
    :caption: examples/tutorials/simple_sine/sine_sim.py

Our simulator function is called by a worker for every work item produced by
the generator. This function calculates the sine of the passed value,
and then returns it so the worker can store the result.
