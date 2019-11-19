Calling Scripts
===============

Below are example calling scripts used to populate specifications for each user
function and libEnsemble before initiating libEnsemble via the primary ``libE()``
call.

Local Sine Tutorial
-------------------

This example is from the Local Sine :doc:`Tutorial<../tutorials/local_sine_tutorial>`,
meant to run with Python's multiprocessing as the primary ``comms`` method.

..  literalinclude:: ../../examples/tutorials/tutorial_calling.py
    :language: python

Balsam Job Controller
---------------------

This example is from the regression tests. It configures MPI as the primary
communication method, and demonstrates registering a user-application with the
libEnsemble job controller to be launched by the job controller within the ``sim_f``.

..  literalinclude:: ../../libensemble/tests/regression_tests/script_test_balsam_hworld.py
    :language: python
