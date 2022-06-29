Output Management
=================

Default Log Files
~~~~~~~~~~~~~~~~~
The history array :ref:`H<datastruct-history-array>` and
:ref:`persis_info<datastruct-persis-info>` dictionary are returned to the user
by libEnsemble.  If libEnsemble aborts on an exception, these structures are
dumped automatically to the respective files:

* ``libE_history_at_abort_<sim_count>.npy``
* ``libE_history_at_abort_<sim_count>.pickle``

where ``sim_count`` is the number of points evaluated. To suppress libEnsemble
from producing these two files, set ``libE_specs['save_H_and_persis_on_abort']`` to ``False``.

Two other libEnsemble files produced by default:

* ``libE_stats.txt``: This contains one-line summaries for each user
  calculation. Each summary is sent by workers to the manager and
  logged as the run progresses.

* ``ensemble.log``: This contains logging output from libEnsemble. The default
  logging level is INFO. In order to gain additional diagnostics, the logging
  level can be set to DEBUG. If this file is not removed, multiple runs will
  append output. Messages at or above MANAGER_WARNING are also copied to stderr
  to alert the user promptly.

To suppress libEnsemble from producing these two files, set ``libE_specs['disable_log_files']`` to ``True``.

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
    logger.set_level('DEBUG')

Logger messages of MANAGER_WARNING level or higher are also displayed through stderr by default.
This boundary can be adjusted as follows::

    from libensemble import logger

    # Only display messages with level >= ERROR
    logger.set_stderr_level('ERROR')

stderr displaying can be effectively disabled by setting the stderr level to CRITICAL.

.. automodule:: logger
  :members:
  :no-undoc-members:

.. _output_dirs:

Working Directories for User Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
libEnsemble features configurable output and working directory structuring for
storing results at every step of a calculation, or directing workers to perform
calculations on separate filesystems or in other directories. This is helpful
for users performing simulations or using high-resource generator functions who
want to take advantage of high-speed scratch spaces or disks, or organize their
I/O by application run.

With these features enabled, each time a worker initiates a user function routine
(``gen_f`` or ``sim_f``) it automatically enters a configurable directory,
either a new directory specific to that worker and function instance or a shared
directory for all workers. Where these directories are created or what files
they contain is configurable through settings in :ref:`libE_specs<datastruct-libe-specs>`.
Defining any compatible settings initiates this system with default settings for
unspecified options. Each setting will be described in detail here:

* ``'sim_dirs_make'``: boolean. Enables per-simulation directories with default
  settings. Directories are labeled in the form ``'sim0-worker1'``, by sim ID
  and initiating worker. Without further configuration, directories are placed
  in the ensemble directory ``./ensemble``, relative to where libEnsemble was
  launched. Default: ``True`` with other sim_dir options enabled. If
  ``False``, all workers will operate within the ensemble directory without
  producing per-simulation directories.

* ``'gen_dirs_make'``: boolean. Enabled per-generator instance directories with
  default settings. Directories are labeled in the form ``'gen1-worker1'``. by
  initiating worker and how many times that worker has initiated the generator.
  These behave similarly to simulation directories. Default: ``True`` with
  other gen_dir options enabled.

* ``'ensemble_dir_path'``: This location, typically referred to as the ensemble
  directory, is where each worker places its calculation directories. If not
  specified, calculation directories are placed in ``./ensemble``, relative to
  where libEnsemble was launched. If ``'sim_dirs_make'`` is ``False``, workers
  initiating simulation instances will run within this directory. This behavior
  is similar when ``'gen_dirs_make'`` is ``False``. On supported systems,
  writing to local-node storage is possible and recommended for increased
  performance.::

      libE_specs['ensemble_dir_path'] = "/scratch/my_ensemble"

* ``'use_worker_dirs'``: boolean. Sorts calculation directories into
  per-worker directories at runtime. Particularly useful for organization when
  running with multiple workers on global scratch spaces or the same node, and
  may produce performance benefits. Default: ``False``.

  Default structure with ``'use_worker_dirs'`` unspecified::

        - /ensemble_dir
            - /sim0-worker1
            - /gen1-worker1
            - /sim1-worker2
            ...

  Structure with ``libE_specs['use_worker_dirs'] = True``::

        - /ensemble_dir
            - /worker1
                - /sim0
                - /gen1
                - /sim4
                ...
            - /worker2
            ...

* ``'sim_dir_copy_files'``: A list of paths for files to copy into simulation
  directories. If ``'sim_dirs_make'`` is False, these files are copied to the
  ensemble directory. If using the :ref:`Executor<executor_index>` to launch an
  application, this may be helpful for copying over configuration files for each
  launch.

* ``'gen_dir_copy_files'``: A list of paths for files to copy into generator
  directories. If ``'gen_dirs_make'`` is False, these files are copied to the
  ensemble directory.

* ``'sim_dir_symlink_files'``: A list of paths for files to symlink into
  simulation directories.

* ``'gen_dir_symlink_files'``: A list of paths for files to symlink into
  generator directories.

* ``'ensemble_copy_back'``: boolean. Instructs the manager to create an empty
  directory where libEnsemble was launched where workers copy back their calculation
  directories when a run concludes or an exception occurs. Especially useful when
  ``'ensemble_dir_path'`` has been set to some scratch space or another temporary
  location. Default: ``False``.

* ``'sim_input_dir'``: A path to a directory to copy for simulation
  directories. This directory and its contents are copied to form the base
  of new simulation directories. If ``'sim_dirs_make'`` is False, this directory's
  contents are copied into the ensemble directory.

* ``'gen_input_dir'``: A path to a directory to copy for generator
  directories. This directory and its contents are copied to form the base
  of new generator directories. If ``'gen_dirs_make'`` is False, this directory's
  contents are copied into the ensemble directory.

See the regression tests ``test_sim_dirs_per_calc.py`` and
``test_use_worker_dirs.py`` for examples of many of these settings.
See ``test_sim_input_dir_option.py`` for examples of using these settings
without simulation-specific directories.

.. note::
  The ``scripts`` directory, in the libEnsemble project root directory,
  contains scripts to compare outputs and create plots based on the ensemble output.

.. include:: ../scripts/readme.rst

.. include:: ../scripts/balsam/readme.rst
