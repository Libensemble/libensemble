Calling Scripts
===============

Below are example calling scripts used to populate specifications for each user
function and libEnsemble before initiating libEnsemble via the primary ``libE()``
call. The primary libEnsemble-relevant portions have been highlighted in each
example. Non-highlighted portions may include setup routines, compilation steps
for user-applications, or output processing.

Local Sine Tutorial
-------------------

This example is from the Local Sine :doc:`Tutorial<../tutorials/local_sine_tutorial>`,
meant to run with Python's multiprocessing as the primary ``comms`` method.

..  literalinclude:: ../../examples/tutorials/tutorial_calling.py
    :language: python
    :linenos:
    :emphasize-lines: 8-28

Balsam Job Controller
---------------------

This example is from the regression tests. It configures MPI as the primary
communication method, and demonstrates registering a user-application with the
libEnsemble job controller to be launched by the job controller within the ``sim_f``.

..  literalinclude:: ../../libensemble/tests/regression_tests/script_test_balsam_hworld.py
    :language: python
    :linenos:
    :emphasize-lines: 27-34, 42-65

6-Hump-Camel Persistent APOSMM
------------------------------

This example is also from the regression tests, and demonstrates configuring a
persistent run via a custom allocation function. Note the use of the
``parse_args()`` and ``save_libE_output()`` convenience functions from the
:doc:`utilities<../utilities>`.

..  literalinclude:: ../../libensemble/tests/regression_tests/test_6-hump_camel_persistent_aposmm_1.py
    :language: python
    :linenos:
    :emphasize-lines: 30, 38-69
