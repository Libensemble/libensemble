.. _datastruct-alloc-specs:

Allocation Specs
================

Allocation function specifications to be set in the user calling script. *Optional*.

Can be constructed and passed to libEnsemble as a Python class or a dictionary. When provided as a Python class,
all data is validated immediately on instantiation.

.. autopydantic_model:: libensemble.specs.AllocSpecs
  :model-show-json: False
  :model-show-config-member: False
  :model-show-config-summary: False
  :model-show-validator-members: False
  :model-show-validator-summary: False
  :field-list-validators: False

.. note::
  * libEnsemble uses the following defaults if the user doesn't provide their own ``alloc_specs``:

  ..  literalinclude:: ../../libensemble/specs.py
      :start-at: alloc_f: Callable = give_sim_work_first
      :end-before: end_alloc_tag
      :caption: Default settings for alloc_specs

  * Users can import and adjust these defaults using:

  ..  code-block:: python

      from libensemble.specs import AllocSpecs
      my_new_alloc = AllocSpecs()
      my_new_alloc.alloc_f = another_function

.. seealso::
  - `test_uniform_sampling_one_residual_at_a_time.py`_ specifies fields
    to be used by the allocation function ``give_sim_work_first`` from
    fast_alloc_and_pausing.py_.

  ..  literalinclude:: ../../libensemble/tests/functionality_tests/test_uniform_sampling_one_residual_at_a_time.py
      :start-at: alloc_specs
      :end-before: end_alloc_specs_rst_tag

.. _test_uniform_sampling_one_residual_at_a_time.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling_one_residual_at_a_time.py
.. _fast_alloc_and_pausing.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/fast_alloc_and_pausing.py
