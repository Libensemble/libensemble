.. _datastruct-exit-criteria:

Exit Criteria
=============

.. warning::

    ExitCriteria as a standalone parameter is **deprecated** as of libEnsemble 2.0
    and will be **removed in 2.1**. Pass exit criteria directly to ``Ensemble.run()`` instead:
    ``ensemble.run(sim_max=100)`` or ``ensemble.run(sim_max=100, wallclock_max=3600)``.

The following criteria (or termination tests) can be used to configure when to stop a workflow.

Can be constructed and passed to libEnsemble as a Python class or a dictionary.

.. autopydantic_model:: libensemble.specs.ExitCriteria
  :model-show-json: False
  :model-show-config-member: False
  :model-show-config-summary: False
  :model-show-validator-members: False
  :model-show-validator-summary: False
  :field-list-validators: False

.. seealso::
  From `test_persistent_aposmm_dfols.py`_.

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_persistent_aposmm_dfols.py
      :start-at: exit_criteria
      :end-before: end_exit_criteria_rst_tag

.. _test_persistent_aposmm_dfols.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_dfols.py
