=============
Release Notes
=============

Below are the notes from all libEnsemble releases.

GitHub issues are referenced, and can be viewed with hyperlinks on the `github releases page`_.

.. _`github releases page`: https://github.com/Libensemble/libensemble/releases

Release 0.9.3
-------------

:Date: October 13, 2022

New capabilities:

* New pair of utilities, `liberegister` and `libesubmit` (based on *PSI/J*), for easily preparing and launching libEnsemble workflows with local comms onto most machines and schedulers. #807
* New persistent support function to cancel sim_ids (`request_cancel_sim_ids`). #880
* `keep_state` option for persistent workers: this lets the manager know that the information being sent is intermediate. #880 

Other enhancements:

* The Executor `manager_poll()` interface now sets consistent flags instead of literal strings. #877
* Some internal modules and the test suite now work on Windows. #869 #888
* Specifying the `num_resource_sets` *libE_specs* option instead of `zero_resource_workers` is now recommended except when using a fixed worker/resource mapping. Use ``persis_info["gen_resources"]`` to assign persistent generator resources (default is zero). #905
* An extraneous warning removed. #903

:Note:

* Tested platforms include Linux, MacOS, Windows, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), Swing (A100 GPU system), Perlmutter (HPE Cray EX with A100 NVIDIA GPUs). For Perlmutter, see "Known issues" below.
* Tested Python versions: (Cpython) 3.7, 3.8, 3.9, 3.10.

:Known issues:

* At time of testing on Perlmutter there was an issue running concurrent applications on a node, following a recent system update. This also affects previous versions of libEnsemble, and is being investigated.
* See known issues section in the documentation.

Release 0.9.2
-------------

:Date: July 06, 2022

New capabilities:

* Support auto-detection of PBS node lists. #602
* Added configuration options for `libE_stats.txt` file. #743
* Support for `spawn` and `forkserver` multiprocessing start methods. #797

 * Note that macOS no longer switches to using `fork`. macOS (since Python 3.8) and Windows default to
   using `spawn`. When using `spawn`, we recommend placing calling script code in an ``if __name__ == "__main__":`` block.
   The multiprocessing interface can be used to switch methods (https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method).

Updates to example functions:

Added simple dynamic sampling example. #833
Added heFFTe example. #844
Regression tests separated into problem examples and functionality tests. #839

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), Swing (A100 GPU system), Perlmutter (HPE Cray EX with A100 NVIDIA GPUs).
* Tested Python versions: (Cpython) 3.7, 3.8, 3.9, 3.10.

:Known issues:

* The APOSMM generator function has been noted to operate slower than expected with the `spawn` multiprocessing start method. For this reason we recommend using `fork` with APOSMM, when using `local` comms (`fork` is the default method on Linux systems).
* See known issues section in the documentation.

Release 0.9.1
-------------

:Date: May 11, 2022

This release has minimal changes, but a large number of touched lines.

* Reformatted code for **black** compliance, including string normalization. #811, #814, #821
* Added Spock and Crusher guides. #802
* User can now set ``calc_status`` to any string (for output in libE_stats). #808
* Added a workflows community initiative file. #817

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), Swing (A100 GPU system), Perlmutter (HPE Cray EX with A100 NVIDIA GPUs).
* Tested Python versions: (Cpython) 3.7, 3.8, 3.9, 3.10.

:Known issues:

* See known issues section in the documentation.

Release 0.9.0
-------------

:Date: Apr 29, 2022

Featured new capabilities:

* New `Balsam` Executor with multi-site capability (run user applications on remote systems). #631, #729
* Support for `funcX` (place user functions on remote systems).  #712 / #713
* Added partial support for concurrent/futures interface. (cancel(), cancelled(), done(), running(), result(), exception() and context manager) #719

Breaking API / helper function changes:

See "Updating for libEnsemble v0.9.0" wiki for details:
https://github.com/Libensemble/libensemble/wiki/Updating-for-libEnsemble-v0.9.0

* Scheduler options moved from `alloc_specs['user']` to `libE_specs`. #790
* `BalsamMPIExecutor` is now `LegacyBalsamMPIExecutor`. #729
* The exit_criteria `elapsed_wallclock_time` has been renamed `wallclock_max`.  #750 (with a deprecation warning)
* Clearer and consistent naming of libE-protected fields in history array. #760

Updates to example functions:

* Moved some examples to new repository - [libe-community-examples](https://github.com/Libensemble/libe-community-examples) (VTMOP, DEAP, DeepDriveMD).  #716,  #721, #726
* Updates to Tasmanian examples to include asynchronous generator example. #727 / #732
* Added multi-task, multi-fidelity optimization regression tests using `ax`. #717 / #720

Other functionality enhancements:

* Non-blocking option added for persistent user function receives. #752
* Added `match_slots` option to resource scheduler. #746

Documentation:

* Added tutorial on assigning tasks to GPUs. #768
* Refactored Executor tutorial for simplicity. #749
* Added Perlmutter guide. #728
* Added Slurm guide. #728
* Refactored examples and tutorials - added exercises. #736 / #737
* Updated history array documentation with visual workflow example. #723

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), Swing (A100 GPU system), Perlmutter (HPE Cray EX with A100 NVIDIA GPUs).
* Tested Python versions: (Cpython) 3.7, 3.8, 3.9, 3.10.

:Known issues:

* OpenMPI does not work with direct MPI job launches in ``mpi4py`` comms mode,
  since it does not support nested MPI launches.
  (Either use local mode or the Balsam Executor.)
* See known issues section in the documentation for more issues.

Release 0.8.0
-------------

:Date: Oct 20, 2021

Featured new capabilities:

* Variable resource workers (dynamic reassignment of resources to workers). #643
* Alternative libE interface. An Ensemble object is created and can be parameterized by a YAML file.  #645
* Improved support classes/functions for alloc/gen/sims and executors.
* Many new example generator/simulators and workflows.

Breaking API / helper function changes:

See "Updating for libEnsemble v0.8.0" wiki for details:
https://github.com/Libensemble/libensemble/wiki/Updating-for-libEnsemble-v0.8.0

* Resources management is now independent of the executor.  #345
* The ``'persis_in'`` field has been added to gen_specs/sim_specs (instead of being hard-coded in alloc funcs). #626 / #670
* ``alloc support`` module is now a class. #643 / #656
* ``gen_support`` module is replaced by Persistent Worker support module (now a class). #609 / #671
* Remove ``libE_`` prefix from the logger. #608
* ``avail_worker_ids`` function should specify ``EVAL_GEN_TAG`` or ``EVAL_SIM_TAG`` instead of ``True``. #615 #643
* Pass ``libE_info`` to allocation functions (allows more flexibility for user and efficiency improvements). #672
* ``'given_back'`` is now a protected libEnsemble field in the manager's history array. #651
* Several name changes to functions and parameters (See the wiki above for details). #529 / #659

Updates to example functions:

* Suite of distributed optimization methods for minimizing sums of convex functions. #647 / #649. Methods include:

 * primal-dual sliding (https://arxiv.org/pdf/2101.00143).
 * N-agent, or distributed gradient descent w/ gradient tracking (https://arxiv.org/abs/1908.11444).
 * proximal sliding (https://arxiv.org/abs/1406.0919).

* Added batched construction for Tasmanian example. #644
* Added Tasmanian dependency to Spack package. spack/spack#25762
* Added VTMOP source code and example usage. #676
* Added a multi-fidelity persistent_gp regression test. #683 / #684
* Added a DeepDriveMD inspired workflow. #630
* Created a persistent sim example. #614 / #615
* Added an example where cancellations are given from the alloc func. #677

Other functionality changes:

* A helper function for generic task polling loop has been added. #572 / #612
* Break main loop now happens when sim_max is returned rather than given out. #624
* Enable a final communication with gen. #620 / #628
* Logging updates - includes timestamps, enhanced debug logging, and libEnsemble version. #629 / #674

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), Swing (A100 GPU system).
* Tested Python versions: (Cpython) 3.6, 3.7, 3.8, 3.9, 3.10 [#]_.

.. [#] A reduced set of tests were run for python 3.10 due to some unavailable test dependencies at time of release.

:Known issues:

* OpenMPI does not work with direct MPI job launches in ``mpi4py`` comms mode,
  since it does not support nested MPI launches.
  (Either use local mode or the Balsam Executor.)
* See known issues section in the documentation for more issues.

Release 0.7.2
-------------

:Date: May 03, 2021

API additions:

* Active receive option added that allows irregular manager/worker communication patterns. (#527 / #595)
* A mechanism is added for the cancellation/killing of previously issued evaluations. (#528 / #595 / #596)
* A submit function is added in the base ``Executor`` class that runs a serial application locally. (#531 / #595)
* Added libEnsemble history array protected fields: `returned_time`, `last_given_time`, and `last_gen_time`. (#590)
* Updated libE_specs options (``mpi_comm`` and ``profile``). (#547 / #548)
* Explicit seeding of random streams in ``add_unique_random_streams()`` is now possible. (#542 / #545)

Updates to example functions:

* Added Surmise calibration generator function and two examples (regression tests). (#595)

Other changes:

* Better support for uneven worker to node distribution (including at sub-node level). (#591 / #600)
* Fixed crash when running on Windows. (#534)
* Fixed crash when running with empty `persis_info`. (#571 / #578)
* Error handling has been made more robust. (#592)
* Improve ``H0`` processing (esp. for pre-generated, but not evaluated points). (#536 / #537)
* A global ``sim_id`` is now given, rather than a local count, in _libE_stats.txt_. Also a global gen count is given. (#587, #588)
* Added support for Python 3.9. (#532 / Removed support for Python 3.5. (#562)
* Improve SLURM nodelist detection (more robust). (#560)
* Add check that user does not change protected history fields (Disable via ``libE_specs['safe_mode'] = False``). (#541)
* Added ``print_fields.py`` script for better interrogating the output history files. (#558)
* In examples, ``is_master`` changed to ``is_manager`` to be consistent with manager/worker nomenclature. (#524)

Documentation:

* Added tutorial **Borehole Calibration with Selective Simulation Cancellation**. (#581 / #595)

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm).
* Tested Python versions: (Cpython) 3.6, 3.7, 3.8, 3.9.

:Known issues:

* OpenMPI does not work with direct MPI job launches in ``mpi4py`` comms mode, since it does not support nested MPI launches
  (Either use local mode or Balsam Executor).
* See known issues section in the documentation for more issues.

Release 0.7.1
-------------

:Date: Oct 15, 2020

Dependencies:

* ``psutils`` is now a required dependency. (#478 #491)

API additions:

* Executor updates:

  * Addition of a zero-resource worker option for persistent gens (does not allocate nodes to gen). (#500)
  * Multiple applications can be registered to the Executor (and submitted) by name. (#498)
  * Wait function added to Tasks. (#499)

* Gen directories can now be created with options analogous to those for sim dirs. (#349 / #489)

Other changes:

* Improve comms efficiency (Repack fields when NumPy version 1.15+). (#511)
* Fix multiprocessing error on macOS/Python3.8 (Use 'fork' instead of 'spawn'). (#502 / #503)

Updates to example functions:

* Allow APOSMM to trigger ensemble exit when condition reached. (#507)
* Improvement in how persistent APOSMM shuts down subprocesses (preventing PETSc MPI-abort). (#478)

Documentation:

* APOSMM Tutorial added. (#468)
* Writing guide for user functions added to docs (e.g., creating sim_f, gen_f, alloc_f). (#510)
* Addition of posters and presentations section to docs (inc. Jupyter notebooks/binder links). (#492 #497)

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), and Bridges (HPE system at PSC).
* Cori (Cray XC40/Slurm) was not tested with release code due to system issues.
* Tested Python versions: (Cpython) 3.5, 3.6, 3.7, 3.8.

:Known issues:

* We currently recommend running in Central mode on Bridges, as distributed runs are experiencing hangs.
* OpenMPI does not work with direct MPI job launches in mpi4py comms mode, since it does not support nested MPI launches
  (Either use local mode or Balsam Executor).
* See known issues section in the documentation for more issues.

Release 0.7.0
-------------

:Date: May 22, 2020

Breaking API changes:

* `Job_controller`/`Job` renamed to `Executor`/`Task` and ``launch`` function to ``submit``. (#285)
* Executors/Resources/Utils moved into sub-packages. ``utils`` now in package ``tools``. (#285)
* sim/gen/alloc support functions moved into ``tools`` sub-package. (#285)
* Restructuring of `sim` directory creation with ``libE_specs`` configuration options.
  E.g: When ``sim_input_dir`` is given, directories for each `sim` are created. (#267)
* User can supply a file called ``node_list`` (replaces ``worker_list``). (#455)

API additions:

* Added gen_funcs.rc configuration framework with option to select APOSMM Optimizers for import. (#444)
* Provide ``alloc_specs`` defaults via `alloc_funcs.defaults` module. (#325)
* Added ``extra_args`` option to the Executor submit function to allow addition of arbitrary MPI runner options. (#445)
* Added ``custom_info`` argument to MPI Executor to allow overriding of detected settings. (#448)
* Added ``libE_specs`` option to disable log files. (#368)

Other changes:

* Added libEnsemble Conda package, hosted on conda-forge.
* Bugfix: Intermittent failures with repeated libE calls under `mpi4py` comms.
  Every libE call now uses its own duplicate of provided communicator and closes out. (#373/#387)
* More accurate timing in `libE_stats.txt`. (#318)
* Addition of new post-processing scripts.

Updates to example functions:

* Persistent APOSMM is now the recommended APOSMM (`aposmm.py` renamed to `old_aposmm.py`). (#435)
* New alloc/gen func: Finite difference parameters with noise estimation.  (#350)
* New example gen func: Tasmanian UQ generator.  (#351)
* New example gen func: Deap/NSGA2 generator.  (#407)
* New example gen func to interface with VTMOP.
* New example sim func: Borehole. (#367)
* New example use-case: WarpX/APOSMM. (#425)

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), Cori (Cray XC40/Slurm), and Bridges (HPE system at PSC).
* Tested Python versions: (Cpython) 3.5, 3.6, 3.7, 3.8.

:Known issues:

* We currently recommended running in Central mode on Bridges as distributed runs are experiencing hangs.
* See known issues section in the documentation for more issues.

Release 0.6.0
-------------

:Date: December 4, 2019

API changes:

* sim/gen/alloc_specs options that do not directly involve these routines are moved to libE_specs (see docs). (#266, #269)
* sim/gen/alloc_specs now require user-defined attributes to be added under the ``'user'`` field (see docs and examples). (#266, #269)
* Addition of a utils module to help users create calling scripts. Includes an argument parser and utility functions. (#308)
* check_inputs() function is moved to the utils module. (#308)
* The libE_specs option ``nprocesses`` has been changed to ``nworkers``. (#235)

New example functions:

* Addition of a persistent APOSMM generator function. (#217)

Other changes:

* Overhaul of documentation, including HPC platform guides and a new pdf structure. (inc. #232, #282)
* Addition of OpenMP threading and GPU support to forces test. (#250)
* Balsam job_controller now tested on Travis. (#47)

:Note:

* Tested platforms include Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), and Cori (Cray XC40/Slurm).
* Tested Python versions: (Cpython) 3.5, 3.6, 3.7

:Known issues:

* These are unchanged from v0.5.0.
* A known issues section has now been added to the documentation.

Release 0.5.2
-------------

:Date: August 19, 2019

* Code has been restructured to meet xSDK package policies for interoperable ECP software (version 0.5.0). #208
* The use of MPI.COMM_WORLD has been removed. Uses a duplicate of COMM_WORLD if no communicator passed (any process not in communicator returns with an exit code of 3). #108
* All output from libEnsemble goes via logger. MANAGER_WARNING level added. This level and above are echoed to stderr by default. API option to change echo level.
* Simulation directories are created only during sim_f calls are suffixed by _worker. #146
* New user function libE.check_inputs() can be used to check valid configuration of inputs. Can be called in serial or under MPI (see libE API). #65
* Installation option has been added to install dependencies used in tests ``pip install libensemble[extras]``.
* A profiling option has been added to sim_specs. #170
* Results comparison scripts have been included for convenience.

:Note:

* Tested platforms include Linux, MacOS (**New**), Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), and Bebop (Cray CS400/Slurm).
* Tested Python versions: (Cpython) 3.5, 3.6, 3.7
* **Note** Support has been removed for Python 3.4 since it is officially retired. Also NumPy has removed support.

:Known issues:

* These are unchanged from v0.5.0.

Release 0.5.1
-------------

:Date: July 11, 2019

* Fixed LSF resource detection for large jobs on LSF systems (e.g., Summit). #184
* Added support for macOS. #182
* Improved the documentation (including addition of beginner's tutorial and FAQ).

:Note:

* Tested platforms include Local Linux, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), and Bebop (Cray CS400/Slurm).
* Tested Python versions: (Cpython) 3.4, 3.5, 3.6, 3.7.

:Known issues:

* These are unchanged from v0.5.0.

Release 0.5.0
-------------

:Date: May 22, 2019

* Added local (multiprocessing) and TCP options for manager/worker communications, in addition to mpi4py. (#42).

 * Example: libEnsemble can be run on MOM/launch nodes (e.g., those of ALCF/Theta & OLCF/Summit) and can remotely detect compute resources.
 * Example: libEnsemble can be run on a system without MPI.
 * Example: libEnsemble can be run with a local manager and remote TCP workers.

* Added support for Summit/LSF scheduler in job controller.
* MPI job controller detects and retries launches on failure; adding resilience. (#143)
* Job controller supports option to extract/print job times in libE_stats.txt. (#136)
* Default logging level changed to INFO. (#164)
* Logging interface added, which allows user to change logging level and file. (#110)
* All worker logging and calculation stats are routed through manager.
* libEnsemble can be run without a gen_func, for example, when using a previously computed random sample. (#122)
* Aborts dump persis_info with the history.

:Note:

* **This version no longer supports Python 2.**
* Tested platforms include Local Linux, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), and Bebop (Cray CS400/Slurm).

:Known issues:

* OpenMPI does not work with direct MPI job launches in mpi4py comms mode, since it does not support nested MPI launches
  (Either use local mode or Balsam job controller).
* Local comms mode (multiprocessing) may fail if MPI is initialized before forking processors. This is thought to be responsible for issues combining with PETSc.
* Remote detection of logical cores via LSB_HOSTS (e.g., Summit) returns number of physical cores since SMT info not available.
* TCP mode does not support (1) more than one libEnsemble call in a given script or (2) the auto-resources option to the job controller.

Release 0.4.1
-------------

:Date: February 20, 2019

* Logging no longer uses root logger (also added option to change libEnsemble log level). (#105)
* Added wait_on_run option for job controller launch to block until jobs have started. (#111)
* persis_info can be passed to sim as well as gen functions. (#112)
* Postprocessing scripts added to create performance/utilization graphs. (#102)
* New scaling test added (not part of current CI test suite). (#114)

Release 0.4.0
-------------

:Date: November 7, 2018

* Separated job controller classes into different modules including a base class (API change).
* Added central_mode run option to distributed type (MPI) job_controllers (API addition). (#93)
* Made poll and kill job methods (API change).
* In job_controller, set_kill_mode is removed and replaced by a wait argument for a hard kill (API change).
* Removed register module - incorporated into job_controller (API change).
* APOSMM has improved asynchronicity when batch mode is false (with new example). (#96)
* Manager errors (instead of hangs) when alloc_f or gen_f don't return work when all workers are idle. (#95)

:Known issues:

* OpenMPI is not supported with direct MPI launches since nested MPI launches are not supported.

Release 0.3.0
-------------

:Date: September 7, 2018

* Issues with killing jobs have been fixed. (#21)
* Fixed job_controller manager_poll to work with multiple jobs. (#62)
* API change: persis_info now included as an argument to libE and is returned from libE instead of gen_info
* Gen funcs: aposmm_logic module renamed to aposmm.
* New example gen and allocation functions.
* Updated Balsam launch script (with new Balsam workflow).
* History is dumped to file on manager or worker exception and MPI aborted (with exit code 1). (#46)
* Default logging level changed to DEBUG and redirected to file ensemble.log.
* Added directory of standalone tests (comms, job kills, and nested MPI launches).
* Improved and speeded up unit tests. (#68)
* Considerable documentation enhancements.

:Known issues:

* OpenMPI is not supported with direct MPI launches since nested MPI launches are not supported.

Release 0.2.0
-------------

:Date: June 29, 2018

* Added job_controller interface (for portable user scripts).
* Added support for using the Balsam job manager. Enables portability and dynamic scheduling.
* Added autodetection of system resources.
* Scalability testing: Ensemble performed with 1023 workers on Theta (Cray XC40) using Balsam.
* Tested MPI libraries: MPICH and Intel MPI.

:Known issues:

* Killing MPI jobs does not work correctly on some systems (including Cray XC40 and CS400). In these cases, libEnsemble continues, but processes remain running.
* OpenMPI does not work correctly with direct launches (and has not been tested with Balsam).

Release 0.1.0
-------------

:Date: November 30, 2017

* Initial release.
