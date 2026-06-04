APOSMM
------

.. autoclass:: gen_classes.aposmm_direct.APOSMM
  :members: suggest, ingest, export, suggest_updates, finalize
  :undoc-members:
  :show-inheritance:


APOSMM with libEnsemble
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../libensemble/tests/regression_tests/test_asktell_aposmm_nlopt.py
    :linenos:
    :start-at:        workflow = Ensemble(parse_args=True)
    :end-before:         # Perform the run

APOSMM standalone
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../libensemble/tests/unit_tests/test_aposmm.py
    :linenos:
    :start-at: def test_aposmm_ingest_first():
    :end-before:    assert exit_code == FINISHED_PERSISTENT_GEN_TAG
