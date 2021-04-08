.. _datastruct-history-array:

history array
=============
::

    H: numpy structured array
        History to store output from gen_f/sim_f/alloc_f for each entry

Fields in ``H`` include those specified in ``sim_specs['out']``,
``gen_specs['out']``, and ``alloc_specs['out']``. All values are initiated to
0 for integers, 0.0 for floats, and False for Booleans.

Below are the protected fields used in ``H``. Other than ``'sim_id'`` and
``cancel_requested``, these fields cannot be overwritten by user functions (unless
``libE_spces['safe_mode']`` is set to ``False``).

..  literalinclude:: ../../libensemble/tools/fields_keys.py
    :start-at: libE_fields
    :end-before: end_libE_fields_rst_tag

.. seealso::

  Example :doc:`sim_specs<./sim_specs>`, :doc:`gen_specs<./gen_specs>`, and :doc:`alloc_specs<./alloc_specs>`.

.. hint::
  Users can check the internal consistency of a history array by importing
  ``check_inputs()`` and calling it with their ``gen_specs``, ``alloc_specs``,
  and ``sim_specs`` as keyword arguments::

      from libensemble.tools import check_inputs

      check_inputs(H0=my_H, sim_specs=sim_specs, alloc_specs=alloc_specs, gen_specs=gen_specs)
