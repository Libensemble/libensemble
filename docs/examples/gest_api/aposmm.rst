APOSMM
------

.. automodule:: gen_classes.aposmm
  :members:
  :undoc-members:
  :show-inheritance:


.. seealso::

   .. tab-set::

    .. tab-item:: APOSMM with libEnsemble

      .. literalinclude:: ../../../libensemble/tests/regression_tests/test_asktell_aposmm_nlopt.py
        :linenos:
        :start-at:        workflow.libE_specs.gen_on_manager = True
        :end-before:         # Perform the run

    .. tab-item:: APOSMM standalone

      .. literalinclude:: ../../../libensemble/tests/unit_tests/test_persistent_aposmm.py
        :linenos:
        :start-at: def test_asktell_ingest_first():
        :end-before:    assert persis_info.get("run_order"), "Standalone persistent_aposmm didn't do any localopt runs"
