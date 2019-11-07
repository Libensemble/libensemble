Programming with libEnsemble
============================

.. automodule:: libE
  :members:
  :no-undoc-members:

**libEnsemble Output**

The history array :ref:`H<datastruct-history-array>` and
:ref:`persis_info<datastruct-persis-info>` dictionary are returned to the user
by libEnsemble.  If libEnsemble aborts on an exception, these structures are
dumped to the respective files,

* ``libE_history_at_abort_<sim_count>.npy``
* ``libE_history_at_abort_<sim_count>.pickle``

where ``sim_count`` is the number of points evaluated.

Other libEnsemble files produced by default are:

* ``libE_stats.txt``: This contains a one-line summary of all user
  calculations.  Each calculation summary is sent by workers to the manager and
  printed as the run progresses.

* ``ensemble.log``: This is the logging output from libEnsemble. The default
  logging is at INFO level. To gain additional diagnostics logging level can be
  set to DEBUG.  If this file is not removed, multiple runs will append output.
  Messages at or above level MANAGER_WARNING are also copied to stderr to alert
  the user promptly.  For more info, see :doc:`Logging<logging>`.

.. toctree::
   data_structures/data_structures
   user_funcs
   job_controller/jc_index
   logging
