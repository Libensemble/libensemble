Output Management
=================

Each of the following output files can be placed in a run-specific
directory by setting ``libE_specs["use_workflow_dir"] = True``.

Default Log Files
~~~~~~~~~~~~~~~~~
The history array :ref:`H<funcguides-history>` and
:ref:`persis_info<datastruct-persis-info>` dictionary are returned to the user
by libEnsemble.  If libEnsemble aborts on an exception, these structures are
dumped automatically to these files:

* ``libE_history_at_abort_<sim_count>.npy``
* ``libE_history_at_abort_<sim_count>.pickle``

To suppress libEnsemble from producing these two files, set ``libE_specs["save_H_and_persis_on_abort"] = False``.

Two other libEnsemble files produced by default:

* ``libE_stats.txt``: One-line summaries for each user calculation.

* ``ensemble.log``: Logging output. Multiple runs will append output if this file isn't removed. See below for config info.

To suppress libEnsemble from producing these two files, set ``libE_specs["disable_log_files"] = True``.

.. _logger_config:

Logger Configuration
~~~~~~~~~~~~~~~~~~~~

The libEnsemble logger uses the standard Python logging levels (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``)
plus one additional custom level (``MANAGER_WARNING``) between ``WARNING`` and ``ERROR``.

The default level is ``INFO``, which includes information about how tasks are submitted
and when tasks are killed. To gain additional diagnostics, set the logging level
to ``DEBUG``. libEnsemble writes to ``ensemble.log`` by default. A log
file name can also be supplied.

To change the logging level to ``DEBUG``::

    from libensemble import logger
    logger.set_level("DEBUG")

Logger messages of ``MANAGER_WARNING`` level or higher are also displayed through stderr by default.
This boundary can be adjusted::

    from libensemble import logger

    # Only display messages with level >= ERROR
    logger.set_stderr_level("ERROR")

stderr displaying can be effectively disabled by setting the stderr level to ``CRITICAL``.

.. automodule:: logger
  :members:
  :no-undoc-members:

.. note::
  The ``scripts`` directory, in the libEnsemble project root directory,
  contains scripts to compare outputs and create plots based on the ensemble output.

.. include:: ../scripts/readme.rst
