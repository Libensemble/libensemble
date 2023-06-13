.. _datastruct-persis-info:

persis_info
===========

Optionally supply persistent information to libEnsemble::

    persis_info: [dict]:
        Dictionary containing persistent info

Holds data to be passed to and from workers updating some state information.

When using ``multiprocessing`` manager-worker comms, these objects are passed *by reference*.

If worker ``i`` sends back ``persis_info``, it is stored in ``persis_info[i]``.

.. dropdown:: Examples

    1.  Random number generators or other structures for use on consecutive calls
    2.  Incrementing array row indexes or process counts
    3.  Sending/receiving updated models from workers

.. seealso::

  From: support.py_

  ..  literalinclude:: ../../libensemble/tests/regression_tests/support.py
      :start-at: persis_info_1
      :end-before: end_persis_info_rst_tag

.. _support.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/support.py
