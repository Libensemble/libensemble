.. _datastruct-history-array:

history array
=============
::

    H: numpy structured array
        History to store output from gen_f/sim_f/alloc_f for each entry

Fields in ``H`` include those specified in ``sim_specs['out']``,
``gen_specs['out']``, and ``alloc_specs['out']``. All values are initiated to
0 for integers, 0.0 for floats, and False for booleans.

Below are the protected fields used in ``H``

..  literalinclude:: ../../libensemble/utils.py
    :start-at: libE_fields
    :end-before: end_libE_fields_rst_tag

.. seealso::

  See example :doc:`sim_specs<./sim_specs>`, :doc:`gen_specs<./gen_specs>`, and :doc:`alloc_specs<./alloc_specs>`.

.. hint::
  Users can check the internal consistency of a History array by importing
  ``check_inputs()`` and calling it with their ``gen_specs``, ``alloc_specs``,
  and ``sim_specs`` as keyword arguments::

      from libensemble.utils import check_inputs

      check_inputs(H0=my_H, sim_specs=sim_specs, alloc_specs=alloc_specs, gen_specs=gen_specs)
