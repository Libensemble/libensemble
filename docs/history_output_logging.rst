Output Management
=================

Each of the following described output files and directories can be placed in a run-specific
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

.. _output_dirs:

Working Directories for User Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By setting certain options in :ref:`libE_specs<datastruct-libe-specs>`,
libEnsemble can direct workers to call their user functions on separate filesystems
or in other directories. This is helpful for taking advantage of scratch spaces or
organizing I/O  by application run.

* ``"sim_dirs_make"``: ``[bool] = False``. Enables per-simulation directories with default
  settings. Directories are placed in ``ensemble`` by default.

* ``"gen_dirs_make"``: ``[bool] = False``. Enabled per-generator instance directories with
  default settings. Directories are placed in ``ensemble`` by default.

* ``"ensemble_dir_path"``: ``[str] = "./ensemble"``. Specifies where each worker places its
  calculation directories. If ``"sim_dirs_make"`` or ``"gen_dirs_make"`` are ``False`` respectively,
  matching workers will run within this directory::

      libE_specs["ensemble_dir_path"] = "/scratch/my_ensemble"

* ``"use_worker_dirs"``: ``[bool] = False``. Sorts calculation directories into
  per-worker directories at runtime. Particularly useful for organizing when
  running with multiple workers on global scratch spaces or the same node, and
  may produce performance benefits.

  Default structure with ``"use_worker_dirs"`` unspecified::

        - /ensemble_dir
            - /sim0-worker1
            - /gen1-worker1
            - /sim1-worker2
            ...

  Structure with ``libE_specs["use_worker_dirs"] = True``::

        - /ensemble_dir
            - /worker1
                - /sim0
                - /gen1
                - /sim4
                ...
            - /worker2
            ...

* ``"sim_dir_copy_files"``: ``[List[str]] = []``. A list of paths to files to copy into simulation
  directories. If ``"sim_dirs_make"`` is ``False``, these files are copied into the
  ensemble directory. If using the :ref:`Executor<executor_index>` to launch an
  application, this may be helpful for copying over configuration files for each
  launch.

* ``"gen_dir_copy_files"``: ``[List[str]] = []``. A list of paths for files to copy into generator
  directories. If ``"gen_dirs_make"`` is ``False``, these files are copied to the
  ensemble directory.

* ``"sim_dir_symlink_files"``: ``[List[str]] = []``. A list of paths for files to symlink into
  simulation directories.

* ``"gen_dir_symlink_files"``: ``[List[str]] = []``. A list of paths for files to symlink into
  generator directories.

* ``"ensemble_copy_back"``: ``[bool] = False``. Instructs libEnsemble to copy back calculation
  directories when a run concludes or an exception occurs to the launch location. Especially useful when
  ``"ensemble_dir_path"`` has been set to some scratch space or another temporary
  location.

* ``"sim_input_dir"``: ``[str] = ""``. A path to a directory to copy to create a set of default contents
  for new simulation directories.  If ``"sim_dirs_make"`` is ``False``, this directory's
  contents are copied into the ensemble directory.

* ``"gen_input_dir"``: ``[str] = ""``. A path to a directory to copy to create a set of default contents
  for new generator directories. If ``"gen_dirs_make"`` is ``False``, this directory's
  contents are copied into the ensemble directory.

See the regression tests ``test_sim_dirs_per_calc.py`` and
``test_use_worker_dirs.py`` for examples of many of these settings.
See ``test_sim_input_dir_option.py`` for examples of using these settings
without simulation-specific directories.

.. note::
  The ``scripts`` directory, in the libEnsemble project root directory,
  contains scripts to compare outputs and create plots based on the ensemble output.

.. include:: ../scripts/readme.rst
