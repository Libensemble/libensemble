Output Management
=================

Each of the following described output files can be placed in a run-specific
directory by setting ``libE_specs["use_workflow_dir"] = True``.

Default Log Files
~~~~~~~~~~~~~~~~~
The history array :ref:`H<funcguides-history>` and
:ref:`persis_info<datastruct-persis-info>` dictionary are returned to the user
by libEnsemble.  If libEnsemble aborts on an exception, these structures are
dumped automatically to the respective files:

* ``libE_history_at_abort_<sim_count>.npy``
* ``libE_history_at_abort_<sim_count>.pickle``

where ``sim_count`` is the number of points evaluated. To suppress libEnsemble
from producing these two files, set ``libE_specs["save_H_and_persis_on_abort"]`` to ``False``.

Two other libEnsemble files produced by default:

* ``libE_stats.txt``: This contains one-line summaries for each user
  calculation. Each summary is sent by workers to the manager and
  logged as the run progresses.

* ``ensemble.log``: This contains logging output from libEnsemble. The default
  logging level is INFO. In order to gain additional diagnostics, the logging
  level can be set to DEBUG. If this file is not removed, multiple runs will
  append output. Messages at or above MANAGER_WARNING are also copied to stderr
  to alert the user promptly.

To suppress libEnsemble from producing these two files, set ``libE_specs["disable_log_files"]`` to ``True``.

.. _logger_config:

Logger Configuration
~~~~~~~~~~~~~~~~~~~~

The libEnsemble logger uses the standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
plus one additional custom level (MANAGER_WARNING) between WARNING and ERROR.

The default level is INFO, which includes information about how tasks are submitted
and when tasks are killed. To gain additional diagnostics, set the logging level
to DEBUG. libEnsemble produces logging to the file ``ensemble.log`` by default. A log
file name can also be supplied.

To change the logging level to DEBUG, provide the following in the calling scripts::

    from libensemble import logger
    logger.set_level("DEBUG")

Logger messages of MANAGER_WARNING level or higher are also displayed through stderr by default.
This boundary can be adjusted as follows::

    from libensemble import logger

    # Only display messages with level >= ERROR
    logger.set_stderr_level("ERROR")

stderr displaying can be effectively disabled by setting the stderr level to CRITICAL.

.. automodule:: logger
  :members:
  :no-undoc-members:

.. note::
  The ``scripts`` directory, in the libEnsemble project root directory,
  contains scripts to compare outputs and create plots based on the ensemble output.

.. include:: ../scripts/readme.rst
