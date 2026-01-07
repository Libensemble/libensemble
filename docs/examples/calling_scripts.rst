Calling Scripts
===============

Below are example calling scripts used to populate specifications for each user
function and libEnsemble before initiating libEnsemble via the primary ``libE()``
call. The primary libEnsemble-relevant portions have been highlighted in each
example. Non-highlighted portions may include setup routines, compilation steps
for user applications, or output processing. The first two scripts correspond to
random sampling calculations, while the third corresponds to an optimization routine.

Many other examples of calling scripts can be found in libEnsemble's `regression tests`_.

Local Sine Tutorial
-------------------

This example is from the Local Sine :doc:`Tutorial<../tutorials/local_sine_tutorial>`,
meant to run with Python's multiprocessing as the primary ``comms`` method.

..  literalinclude:: ../../examples/tutorials/simple_sine/test_local_sine_tutorial.py
    :language: python
    :caption: examples/tutorials/simple_sine/test_local_sine_tutorial.py
    :linenos:
    :emphasize-lines: 8-28

Electrostatic Forces with Executor
----------------------------------

These examples are from a test for evaluating the scaling capabilities of libEnsemble
by calculating particle electrostatic forces through a user application. This
application is registered with the MPIExecutor, then submitted
for execution in the ``sim_f``. Note the use of the ``parse_args=True`` which allows
reading arguments such as the number of workers from the command line.

Traditional Version
~~~~~~~~~~~~~~~~~~~

Run using five workers with::

    python run_libe_forces.py -n 5

One worker runs a persistent generator and the other four run the forces simulations.

..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/forces_simple/run_libe_forces.py
    :language: python
    :caption: tests/scaling_tests/forces/forces_simple/run_libe_forces.py
    :linenos:

Object + yaml Version
~~~~~~~~~~~~~~~~~~~~~

..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/forces_adv/run_libe_forces_from_yaml.py
    :language: python
    :caption: tests/scaling_tests/forces/forces_adv/run_libe_forces_from_yaml.py
    :linenos:

..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/forces_adv/forces.yaml
    :language: yaml
    :caption: tests/scaling_tests/forces/forces_adv/forces.yaml
    :linenos:

Persistent APOSMM with Gradients
--------------------------------

This example is also from the regression tests and demonstrates configuring a
persistent run via a custom allocation function.

..  literalinclude:: ../../libensemble/tests/regression_tests/test_persistent_aposmm_with_grad.py
    :language: python
    :caption: tests/regression_tests/test_persistent_aposmm_with_grad.py
    :linenos:

.. _regression tests: https://github.com/Libensemble/libensemble/tree/develop/libensemble/tests/regression_tests
