.. _examples-alloc:

Allocation Functions
====================

Below are example allocation functions available in libEnsemble.

Many users use these unmodified.

.. IMPORTANT::
  See the API for allocation functions :ref:`here<api_alloc_f>`.

.. note::
   The default allocation function (for non-persistent generators) is :ref:`give_sim_work_first<gswf_label>`.

   The most commonly used (for persistent generators) is :ref:`start_only_persistent<start_only_persistent_label>`.

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
.. automodule:: start_persistent_local_opt_gens
  :members:
  :undoc-members:

fast_alloc_and_pausing
----------------------
.. automodule:: fast_alloc_and_pausing
   :members:
   :undoc-members:

only_one_gen_alloc
------------------
.. automodule:: only_one_gen_alloc
   :members:
   :undoc-members:

start_fd_persistent
-------------------
.. automodule:: start_fd_persistent
   :members:
   :undoc-members:

persistent_aposmm_alloc
-----------------------
.. automodule:: persistent_aposmm_alloc
   :members:
   :undoc-members:

give_pregenerated_work
----------------------
.. automodule:: give_pregenerated_work
   :members:
   :undoc-members:

inverse_bayes_allocf
--------------------
.. automodule:: inverse_bayes_allocf
   :members:
   :undoc-members:
