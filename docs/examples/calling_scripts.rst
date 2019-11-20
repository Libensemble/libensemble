Calling Scripts
===============

Below are example calling scripts used to populate specifications for each user
function and libEnsemble before initiating libEnsemble via the primary ``libE()``
call. The primary libEnsemble-relevant portions have been highlighted in each
example. Non-highlighted portions may include setup routines, compilation steps
for user-applications, or output processing. The first two scripts correspond to
random sampling calculations, while the third corresponds to an optimization routine.

Local Sine Tutorial
-------------------

This example is from the Local Sine :doc:`Tutorial<../tutorials/local_sine_tutorial>`,
meant to run with Python's multiprocessing as the primary ``comms`` method.

..  literalinclude:: ../../examples/tutorials/tutorial_calling.py
    :language: python
    :caption: examples/tutorials/tutorial_calling.py
    :linenos:
    :emphasize-lines: 8-28

Electrostatic Forces with Job Controller
----------------------------------------

This example is from a test for evaluating the scaling capabilities of libEnsemble
by calculating particle electrostatic forces through a user-application. This
application is registered with either the MPI or Balsam job controller, then
launched in the ``sim_f``. Note the use of the ``parse_args()`` and
``save_libE_output()`` convenience functions from the :doc:`utilities<../utilities>`.

..  literalinclude:: ../../libensemble/tests/scaling_tests/forces/run_libe_forces.py
    :language: python
    :caption: tests/scaling_tests/forces/run_libe_forces.py
    :linenos:
    :emphasize-lines: 16, 39-92

6-Hump-Camel Persistent APOSMM
------------------------------

This example is also from the regression tests, and demonstrates configuring a
persistent run via a custom allocation function.

..  literalinclude:: ../../libensemble/tests/regression_tests/test_6-hump_camel_persistent_aposmm_1.py
    :language: python
    :caption: tests/regression_tests/test_6-hump_camel_persistent_aposmm_1.py
    :linenos:
    :emphasize-lines: 29, 42-72, 85
