.. _datastruct-worker-array:

worker array
=============
::

    W: numpy structured array
        'active' [int]:
            Is the worker active or not
        'persis_state' [int]:
            Is the worker in a persis_state
        'blocked' [int]:
            Is the worker's resources blocked by another calculation

The worker array ``W`` contains information about each worker's state. This is
useful information for allocation functions determining what work should be
performed next.

We take the following convention:

=========================================   =======  ============  =======
Worker state                                 active  persis_state  blocked
=========================================   =======  ============  =======
idle worker                                    0          0           0
active, nonpersistent sim                      1          0           0
active, nonpersistent gen                      2          0           0
active, persistent sim                         1          1           0
active, persistent gen                         2          2           0
waiting, persistent sim                        0          1           0
waiting, persistent gen                        0          2           0
worker blocked by some other calculation       1          0           1
=========================================   =======  ============  =======

.. note::
  * libE only receives from workers with a nonzero 'active' state
  * libE only calls the alloc_f if some worker has an 'active' state of zero

.. seealso::
  For an example allocation function that queries the worker array, see
  `persistent_aposmm_alloc`_.

.. _persistent_aposmm_alloc: https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/persistent_aposmm_alloc.py
