.. _datastruct-persis-info:

persis_info
===========

Supply persistent information to libEnsemble::

    persis_info: [dict]:
        Dictionary containing persistent info

Holds data that is passed to and from workers updating some state information. A typical example
is a random number generator to be used in consecutive calls to a generator.

If worker ``i`` sends back ``persis_info``, it is stored in ``persis_info[i]``. This functionality
can be used to, for example, pass a random stream back to the manager to be included in future work
from the allocation function.

.. seealso::

  From: support.py_

  ..  literalinclude:: ../../libensemble/tests/regression_tests/support.py
      :start-at: persis_info_1
      :end-before: end_persis_info_rst_tag

.. _support.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/support.py
