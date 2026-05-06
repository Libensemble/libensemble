Top-Level Scripts
=================

Many other examples of top-level scripts can be found in libEnsemble's `regression tests`_.

Local Sine Tutorial
-------------------

This example is from the Local Sine :doc:`Tutorial<../tutorials/local_sine_tutorial/local_sine_tutorial>`,
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

APOSMM with a Standardized Generator
--------------------------------------

This example from the regression tests demonstrates the v2.0 gest-api interface:
a standardized ``APOSMM`` generator class parameterized by a ``VOCS`` object,
paired with a gest-api ``simulator`` callable. The generator runs on the manager
thread by default, leaving all workers available for simulations.

..  literalinclude:: ../../libensemble/tests/regression_tests/test_asktell_aposmm_nlopt.py
    :language: python
    :caption: tests/regression_tests/test_asktell_aposmm_nlopt.py
    :linenos:
    :end-at: workflow.exit_criteria = ExitCriteria(sim_max=2000)

.. _regression tests: https://github.com/Libensemble/libensemble/tree/develop/libensemble/tests/regression_tests
