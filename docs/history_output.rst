The History Array
~~~~~~~~~~~~~~~~~
libEnsemble uses a NumPy structured array :ref:`H<datastruct-history-array>` to
store corresponding output from each ``gen_f`` and ``sim_f``. Similarly,
``gen_f`` and ``sim_f`` are expected to return output as NumPy structured
arrays. The names of the input fields for ``gen_f`` and ``sim_f``
must be output from ``gen_f`` or ``sim_f``. In addition to the user-function output fields,
the final history from libEnsemble will includes the following:

* ``sim_id`` [int]: Each unit of work output from ``gen_f`` must have an
  associated ``sim_id``. The generator can assign this, but users must be
  careful to ensure points are added in order. For example, if ``alloc_f``
  allows for two ``gen_f`` instances to be running simultaneously, ``alloc_f``
  should ensure that both don’t generate points with the same ``sim_id``.

* ``given`` [bool]: Has this ``gen_f`` output been given to a libEnsemble
  worker to be evaluated yet?

* ``given_time`` [float]: At what time (since the epoch) was this ``gen_f``
  output given to a worker?

* ``sim_worker`` [int]: libEnsemble worker that output was given to for evaluation

* ``gen_worker`` [int]: libEnsemble worker that generated this ``sim_id``

* ``gen_time`` [float]: At what time (since the epoch) was this entry (or
  collection of entries) put into ``H`` by the manager

* ``returned`` [bool]: Has this worker completed the evaluation of this unit of
  work?

Output
~~~~~~
The history array :ref:`H<datastruct-history-array>` and
:ref:`persis_info<datastruct-persis-info>` dictionary are returned to the user
by libEnsemble.  If libEnsemble aborts on an exception, these structures are
dumped to the respective files,

* ``libE_history_at_abort_<sim_count>.npy``
* ``libE_history_at_abort_<sim_count>.pickle``

where ``sim_count`` is the number of points evaluated.

Other libEnsemble files produced by default are:

* ``libE_stats.txt``: This contains one-line summaries for each user
  calculation. Each summary is sent by workers to the manager and
  logged as the run progresses.

* ``ensemble.log``: This contains logging output from libEnsemble. The default
  logging level is INFO. To gain additional diagnostics, the logging level can be
  set to DEBUG. If this file is not removed, multiple runs will append output.
  Messages at or above MANAGER_WARNING are also copied to stderr to alert
  the user promptly. For more info, see :doc:`Logging<logging>`.

Output and Working-Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
libEnsemble features configurable output and working-directory structuring for
flexibly storing results at every step of a calculation, or directing workers to
perform calculations on separate filesystems. This is helpful for users
performing I/O-heavy simulations who may want to take advantage of separate
high-speed scratch spaces or disks.

Each time a worker begins a simulation routine, libEnsemble copies a
specified input directory and its contents to a new location. The worker
will then perform its work inside this new directory. How these directories are
copied or labeled is configurable through settings
in :ref:`libE_specs<datastruct-libe-specs>`. Each setting will be described in
detail here:

* ``'sim_input_dir'``: The directory to be copied. Specify this
  option to enable these features::

    libE_specs['sim_input_dir'] = './my_input'


* ``'ensemble_dir'``: Where to write directory copies. If not specified, writes
  directories to a new directory named ``ensemble`` in the current working
  directory::

      libE_specs['ensemble_dir'] = '/scratch/current_run/my_ensemble'


* ``'use_worker_dirs'``: Boolean. If enabled, libEnsemble also creates
  per-worker directories to store the directories use by each worker.
  Particularly useful for organization when running with potentially hundreds
  of workers or simulations with many steps, and may produce performance benefits.

    Default organization with ``'use_worker_dirs'`` unspecified::

        - /my_ensemble
            - /sim0-worker1
            - /sim1-worker2
            - /sim2-worker3
            - /sim3-worker4
            - /sim4-worker1
            - /sim5-worker2
            ...

    Organization with ``libE_specs['use_worker_dirs'] = True``::

        - /my_ensemble
            - /worker1
                - /sim0
                - /sim4
                - /sim8
                ...
            - /worker2
            - /worker3
            - /worker4
            ...

* ``'copy_input_files'``: A list of filenames to exclusively copy from the input
  directory to each calculation directory. With this setting specified, all other
  files in the input directory will be ignored unless specified in
  ``'symlink_input_files'`` (described next)::

      libE_specs['copy_input_files'] = ['only_copy_this']

* ``'symlink_input_files'``: A list of filenames. Of the files copied from the
  input directory into each calculation directory, create symlinks for these
  files instead of copies::

    libE_specs['symlink_input_files'] = ['symlink_this']

* ``'copy_input_to_parent'``: Boolean. Also copy input-directory contents to
  whichever directories directly contain simulation-directories. By default, this
  is the ensemble directory. If ``'use_worker_dirs'`` is ``True``, then this is
  each worker directory. This also changes the behavior of
  ``'symlink_input_files'`` so calculation-directory symlinks refer to these
  copies instead of the input directory.

* ``'clean_ensemble_dirs'``: Boolean. Following libEnsemble execution clean all
  worker and calculation directories and their contents from the output ensemble
  directory.



See the regression test ``test_worker_sim_dirs.py`` for examples of many of
these settings.

Output Analysis
^^^^^^^^^^^^^^^
The ``postproc_scripts`` directory, in the libEnsemble project root directory,
contains scripts to compare outputs and create plots based on the ensemble output.

.. include:: ../postproc_scripts/readme.rst

.. include:: ../postproc_scripts/balsam/readme.rst
