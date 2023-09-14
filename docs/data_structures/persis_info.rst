.. _datastruct-persis-info:

persis_info
===========

Holds persistent information that can be updated during the ensemble.

An initialized ``persis_info`` dictionary can be provided to the ``libE()`` call
or as an attribute of the ``Ensemble`` class.

Dictionary keys that have an integer value contain entries that are passed to
and from the corresponding workers. These are received in the ``persis_info``
argument of user functions, and returned as the optional second return value.

A typical example is a random number generator stream to be used in consecutive
calls to a generator (see function
:meth:`add_unique_random_streams()<tools.add_unique_random_streams>`)

All other entries persist on the manager and can be updated in the calling script
between ensemble invocations, or in the allocation function.

Examples:

- Random number generators or other structures for use on consecutive calls
- Incrementing array row indexes or process counts
- Sending/receiving updated models from workers
- Keeping track of the number of generators started in an allocation function
- Triggering the shutdown of the ensemble (from the allocation function).

Hint: When there are repeated calls to ``libE()`` or ``ensemble.run()``, users may
need to modify or reset the contents of persis_info in some cases.

.. seealso::

  From: support.py_

  ..  literalinclude:: ../../libensemble/tests/regression_tests/support.py
      :start-at: persis_info_1
      :end-before: end_persis_info_rst_tag

.. _support.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/support.py
