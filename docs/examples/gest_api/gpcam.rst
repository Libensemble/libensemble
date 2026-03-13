gpCAM
------

.. autoclass:: gen_classes.gpCAM.GP_CAM
  :members: suggest, ingest
  :undoc-members:
  :show-inheritance:


.. autoclass:: gen_classes.gpCAM.GP_CAM_Covar
  :members: suggest, ingest
  :undoc-members:
  :show-inheritance:


.. seealso::

   .. literalinclude:: ../../../libensemble/tests/regression_tests/test_asktell_gpCAM.py
     :linenos:
     :start-at:   vocs = VOCS(variables={"x0": [-3, 3], "x1": [-2, 2], "x2": [-1, 1], "x3": [-1, 1]}, objectives={"f": "MINIMIZE"})
     :end-before:        if is_manager:
