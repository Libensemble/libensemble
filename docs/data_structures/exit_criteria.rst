.. _datastruct-exit-criteria:

Exit Criteria
=============

The following criteria (or termination tests) can be used to tell libEnsemble when to stop a given run:

.. autopydantic_model:: libensemble.specs.ExitCriteria
  :model-show-json: False
  :members:

.. seealso::
  From `test_persistent_aposmm_dfols.py`_.

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_persistent_aposmm_dfols.py
      :start-at: exit_criteria
      :end-before: end_exit_criteria_rst_tag

.. _test_persistent_aposmm_dfols.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_dfols.py
