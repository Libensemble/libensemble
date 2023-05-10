.. _datastruct-exit-criteria:

Exit Criteria
=============

The following criteria (or termination tests) can be used to configure when to stop a workflow.

Can be constructed and passed to libEnsemble as a Python class or a dictionary. When provided as a Python class,
all data is validated immediately on instantiation.

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
