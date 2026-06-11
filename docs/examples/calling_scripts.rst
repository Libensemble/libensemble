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

..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/forces_simple/run_libe_forces.py
    :language: python
    :caption: tests/scaling_tests/forces/forces_simple/run_libe_forces.py
    :linenos:

gest-api APOSMM
---------------

This example from the regression tests demonstrates the gest-api interface with a
standardized ``APOSMM`` generator class parameterized by a ``VOCS`` object, and
paired with a gest-api ``simulator`` callable.

..  literalinclude:: ../../libensemble/tests/regression_tests/test_asktell_aposmm_nlopt.py
    :language: python
    :caption: tests/regression_tests/test_asktell_aposmm_nlopt.py
    :linenos:
    :end-at: H, _, _ = workflow.run(sim_max=3000, wallclock_max=600)

.. _regression tests: https://github.com/Libensemble/libensemble/tree/develop/libensemble/tests/regression_tests
