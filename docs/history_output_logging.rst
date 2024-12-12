Output Management
=================

Default Log Files
~~~~~~~~~~~~~~~~~
The history array :ref:`H<funcguides-history>` and
:ref:`persis_info<datastruct-persis-info>` dictionary are returned to the user
by libEnsemble.  If libEnsemble aborts on an exception, these structures are
dumped automatically to these files:

* ``libE_history_at_abort_<sim_count>.npy``
* ``libE_persis_info_at_abort_<sim_count>.pickle``

Two other libEnsemble files produced by default:

* ``libE_stats.txt``: One-line summaries for each user calculation.

* ``ensemble.log``: Logging output. Multiple runs will append output if this file isn't removed. See below for config info.

**Global options:**

``libE_specs["disable_log_files"] = True``: Disable output files

``libE_specs["use_workflow_dir"] = True``: Place output files in workflow-instance directories

``libE_specs["save_H_and_persis_on_abort"] = False``: Disable dumping the History array and ``persis_info`` to files

.. code-block:: python

  from libensemble.specs import LibeSpecs

  specs = LibeSpecs(disable_log_files=True, save_H_and_persis_on_abort=False)

.. _logger_config:

Logger Configuration
~~~~~~~~~~~~~~~~~~~~

The libEnsemble logger uses the following logging levels
(``VDEBUG``, ``DEBUG``, ``INFO``, ``WARNING``, ``MANAGER_WARNING``, ``ERROR``, ``CRITICAL``)

The default level is ``INFO``, which includes information about how tasks are submitted
and when tasks are killed. To gain additional diagnostics, including communication
tracking, set the logging level to ``DEBUG``. In rare cases, the ``VDEBUG`` level may
be useful, which also tracks log messages.

libEnsemble writes to ``ensemble.log`` by default. A log file name can also be supplied.

To change the logging level to ``DEBUG``::

    from libensemble import logger
    logger.set_level("DEBUG")

Logger messages of ``MANAGER_WARNING`` level or higher are also displayed through stderr by default.
This boundary can be adjusted::

    from libensemble import logger

    # Only display messages with level >= ERROR
    logger.set_stderr_level("ERROR")

stderr displaying can be effectively disabled by setting the stderr level to ``CRITICAL``.

.. dropdown:: Logger Module

  .. automodule:: logger
    :members:
    :no-undoc-members:

.. note::
  The ``scripts`` directory, in the libEnsemble project root directory,
  contains scripts to compare outputs and create plots based on the ensemble output.

Analysis Utilities
~~~~~~~~~~~~~~~~~~

.. dropdown:: Analysis Utilities

  .. include:: ../scripts/readme.rst
