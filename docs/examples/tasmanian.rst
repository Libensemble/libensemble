persistent_tasmanian
--------------------

Required: Tasmanian_, pypackaging_, scikit-build_

Example usage: batched_, async_

Note that Tasmanian can be pip installed, but currently must
use either *venv* or *--user* install.

``E.g: pip install scikit-build packaging Tasmanian --user``

.. automodule:: persistent_tasmanian
  :members: sparse_grid_batched, sparse_grid_async
  :undoc-members:

.. role:: underline
    :class: underline

.. dropdown:: :underline:`persistent_tasmanian.py`

   .. literalinclude:: ../../libensemble/gen_funcs/persistent_tasmanian.py
      :language: python
      :linenos:

.. _pypackaging: https://pypi.org/project/pypackaging/
.. _scikit-build: https://scikit-build.readthedocs.io/en/latest/index.html

.. Caution - tasmanian_ is named in example docstring so must be named differently
.. _batched: https://github.com/Libensemble/libensemble/blob/main/libensemble/tests/regression_tests/test_persistent_tasmanian.py
.. _async: https://github.com/Libensemble/libensemble/blob/main/libensemble/tests/regression_tests/test_persistent_tasmanian_async.py
