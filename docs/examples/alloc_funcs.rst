.. _examples-alloc:

Allocation Functions
====================

Below are example allocation functions available in libEnsemble.

.. IMPORTANT::
  See the API for allocation functions :ref:`here<api_alloc_f>`.

.. note::
   The default allocation function is ``give_sim_work_first``.

.. role:: underline
    :class: underline

.. _gswf_label:

give_sim_work_first
-------------------
.. automodule:: give_sim_work_first
  :members:
  :undoc-members:

.. dropdown:: :underline:`give_sim_work_first.py`

   .. literalinclude:: ../../libensemble/alloc_funcs/give_sim_work_first.py
      :language: python
      :linenos:

fast_alloc
----------
.. automodule:: fast_alloc
  :members:
  :undoc-members:

.. dropdown:: :underline:`fast_alloc.py`

   .. literalinclude:: ../../libensemble/alloc_funcs/fast_alloc.py
      :language: python
      :linenos:

.. _start_only_persistent_label:

start_only_persistent
---------------------
.. automodule:: start_only_persistent
  :members:
  :undoc-members:

.. dropdown:: :underline:`start_only_persistent.py`

   .. literalinclude:: ../../libensemble/alloc_funcs/start_only_persistent.py
      :language: python
      :linenos:

start_persistent_local_opt_gens
-------------------------------
.. automodule:: libensemble.alloc_funcs.start_persistent_local_opt_gens
  :members:
  :undoc-members:
