=============
Release Notes
=============

Release 0.6.0
-------------

:Date: December 4, 2019

API changes:

* sim/gen/alloc_specs options that do not directly involve these routines are moved to libE_specs (see docs) (#266, #269)
* sim/gen/alloc_specs now require user-defined attributes to be added under the 'user' field (see docs and examples) (#266, #269).
* Addition of a utils module to help users create calling scripts. Includes an argument parser and utility functions (#308).
* check_inputs() function is moved to the utils module (#308).
* The libE_specs option ``nprocesses`` has been changed to ``nworkers`` (#235)

New example functions:

* Addition of a persistent APOSMM generator function (#217).

Other changes:

* Overhaul of documentation, including HPC platform guides and a new pdf structure (inc. #232, #282)
* Addition of OpenMP threading and GPU support to forces test (#250).
* Balsam job_controller now tested on Travis (#47)

:Note:

* Tested platforms include: Linux, MacOS, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm), Cori (Cray XC40/Slurm).
* Tested Python versions: (Cpython) 3.5, 3.6, 3.7

:Known issues:

* These are unchanged from v0.5.0
* A known issues section has now been added to the documentation.

Release 0.5.2
-------------

:Date: August 19, 2019

* Code restructured to meet xSDK package policies for interoperable ECP software (version 0.5.0). #208
* Removed use of MPI.COMM_WORLD. Uses a duplicate of COMM_WORLD if no communicator passed (any process not in communicator returns with an exit code of 3). #108
* All output from libEnsemble goes via logger. MANAGER_WARNING level added. This level and above are echoed to stderr by default. API option to change echo level.
* Simulation directories are only created as required and are suffixed by _worker. #146
* New user function libE.check_inputs() can be used to check valid configuration of inputs. Can be called in serial or under MPI (See libE API). #65
* Installation option added to install dependencies used in tests ``pip install libensemble[extras]``
* A profiling option has been added to sim_specs. #170
* Results comparison scripts have been included for convenience.

:Note:

* Tested platforms include: Linux, MacOS (**New**), Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm).
* Tested Python versions: (Cpython) 3.5, 3.6, 3.7
* **Note** Support has been removed for Python 3.4 as it is officially retired. Also NumPy has removed support.

:Known issues:

* These are unchanged from v0.5.0

Release 0.5.1
-------------

:Date: July 11, 2019

* Fixed LSF resource detection for large jobs on LSF systems (e.g. Summit) #184
* Added support for macOS #182
* Documentation has been improved (including addition of beginner's tutorial and FAQ).

:Note:

* Tested platforms include: Local Linux, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm).
* Tested Python versions: (Cpython) 3.4, 3.5, 3.6, 3.7

:Known issues:

* These are unchanged from v0.5.0

Release 0.5.0
-------------

:Date: May 22, 2019

* Added local (multiprocessing) and TCP options for manager/worker communications, in addition to mpi4py (#42).

 * E.g., libEnsemble can be run on MOM/launch nodes (e.g., those of ALCF/Theta & OLCF/Summit) and can remotely detect compute resources.
 * E.g., libEnsemble can be run on a system without MPI.
 * E.g., libEnsemble can be run with a local manager and remote TCP workers.

* Added support for Summit/LSF scheduler in job controller.
* MPI job controller detects and re-tries launches on failure; adding resilience (#143).
* Job controller supports option to extract/print job times in libE_stats.txt (#136).
* Default logging level changed to INFO (#164).
* Logging interface added, which allows user to change logging level and file (#110).
* All worker logging and calculation stats are routed through manager.
* libEnsemble can be run without a gen_func, for example, when using a previously computed random sample (#122).
* Aborts dump persis_info with the history.

:Note:

* **This version no longer supports Python 2.**
* Tested platforms include: Local Linux, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm).

:Known issues:

* OpenMPI does not work with direct MPI job launches in mpi4py comms mode, as it does not support nested MPI launches
  (Either use local mode or Balsam job controller).
* Local comms mode (multiprocessing) may fail if MPI is initialized before forking processors. This is thought to be responsible for issues combining with PETSc.
* Remote detection of logical cores via LSB_HOSTS (e.g., Summit) returns number of physical cores as SMT info not available.
* TCP mode does not support: 1) more than one libEnsemble call in a given script or 2) the auto-resources option to the job controller.

Release 0.4.1
-------------

:Date: February 20, 2019

* Logging no longer uses root logger (Also added option to change libEnsemble log level) (#105)
* Added wait_on_run option for job controller launch to block until jobs have started (#111)
* persis_info can be passed to sim as well as gen functions (#112)
* Post-processing scripts added to create performance/utilization graphs (#102)
* New scaling test added (not part of current CI test suite) (#114)

Release 0.4.0
-------------

:Date: November 7, 2018

* Separate job controller classes into different modules including a base class (API change)
* Add central_mode run option to distributed type (MPI) job_controllers (API addition) (#93)
* Make poll and kill job methods (API change)
* In job_controller, set_kill_mode is removed and replaced by a wait argument for a hard kill (API change)
* Removed register module - incorporated into job_controller (API change)
* APOSMM has improved asynchronicity when batch mode is false (with new example). (#96)
* Manager errors (instead of hangs) when alloc_f or gen_f don't return work when all workers are idle. (#95)

:Known issues:

* OpenMPI is not supported with direct MPI launches as nested MPI launches are not supported.

Release 0.3.0
-------------

:Date: September 7, 2018

* Issues with killing jobs have been fixed (#21)
* Fix to job_controller manager_poll to work with multiple jobs (#62)
* API change: persis_info now included as an argument to libE and is returned from libE instead of gen_info
* Gen funcs: aposmm_logic module renamed to aposmm.
* New example gen and allocation functions.
* Updated Balsam launch script (with new Balsam workflow)
* History is dumped to file on manager or worker exception and MPI aborted (with exit code 1) (#46)
* Default logging level changed to DEBUG and redirected to file ensemble.log
* Added directory of standalone tests (comms, job kills, and nested MPI launches)
* Improved and speeded up unit tests (#68)
* Considerable documentation enhancements

:Known issues:

* OpenMPI is not supported with direct MPI launches as nested MPI launches are not supported.

Release 0.2.0
-------------

:Date: June 29, 2018

* Added job_controller interface (for portable user scripts).
* Added support for using the Balsam job manager. Enables portability and dynamic scheduling.
* Added auto-detection of system resources.
* Scalability testing: Ensemble performed with 1023 workers on Theta (Cray XC40) using Balsam.
* Tested MPI libraries: MPICH, Intel MPI.

:Known issues:

* Killing MPI jobs does not work correctly on some systems (including Cray XC40 and CS400). In these cases, libEnsemble continues, but processes remain running.
* OpenMPI does not work correctly with direct launches (and has not been tested with Balsam).

Release 0.1.0
-------------

:Date: November 30, 2017

* Initial Release.
