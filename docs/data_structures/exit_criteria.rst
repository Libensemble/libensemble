.. _datastruct-exit-criteria:

Exit Criteria
=============

The following criteria (or termination tests) can be used to configure when to stop a workflow.

Can be constructed and passed to libEnsemble as a Python class or a dictionary. When provided as a Python class,
all data is validated immediately on instantiation. When provided as a dictionary, all data is validated
upon passing into :meth:`libE()<libensemble.libE.libE>`.

.. autopydantic_model:: libensemble.specs.ExitCriteria
  :model-show-json: False
  :members:

.. seealso::
  From `test_persistent_aposmm_dfols.py`_.

  ..  literalinclude:: ../../libensemble/tests/regression_tests/test_persistent_aposmm_dfols.py
      :start-at: exit_criteria
      :end-before: end_exit_criteria_rst_tag

.. _test_persistent_aposmm_dfols.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_dfols.py
