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

Output Analysis
^^^^^^^^^^^^^^^
The ``postproc_scripts`` directory, in the libEnsemble project root directory,
contains scripts to compare outputs and create plots based on the ensemble output.

.. include:: ../postproc_scripts/readme.rst

.. include:: ../postproc_scripts/balsam/readme.rst
