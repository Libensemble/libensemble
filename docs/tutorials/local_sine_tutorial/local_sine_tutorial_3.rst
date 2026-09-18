3. Simulator
============

:doc:`Introduction <local_sine_tutorial>` || :doc:`1. Getting started <local_sine_tutorial_1>` || :doc:`2. Generator <local_sine_tutorial_2>` || **3. Simulator** || :doc:`4. Script <local_sine_tutorial_4>` || :doc:`5. Next steps <local_sine_tutorial_5>`

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
