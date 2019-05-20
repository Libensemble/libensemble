=============
Release Notes
=============


Release 0.5.0
-------------

:Date: May 20, 2019

* Added local(multiprocessing) and TCP options for manager/worker communications, in addition to mpi4py (#42).
  * E.g. libEnsemble can be run on MOM/launch nodes of (e.g. Theta/Summit) and can remotely detect compute resources.
  * E.g. libEnsemble can be run on a system without MPI.
  * E.g. libEnsemble can be run with a manager locally and remote TCP workers.

* Added support for Summit/LSF schedular in job controller.
* MPI Job controller detects and re-tries launches on failure; adding resilience (#143).
* Job controller supports option to extract/print job times in libE_stats.txt (#136)
  (Plotting script added to show resource utilization for launched jobs).

* Default logging level changed to INFO (#164)
* Logging interface added which allows user to change logging level and file (#110).
* All worker logging and calculation stats are routed through manager.
* Simple libE use cases should not require a gen_func (e.g. random sampling) (#122).

:Note:

* **This version no longer supports Python2.** From this version, libEnsemble requires Python 3.

* Tested platforms include: Local Linux, Theta (Cray XC40/Cobalt), Summit (IBM Power9/LSF), Bebop (Cray CS400/Slurm).

:Known limitations:

* OpenMPI does work with direct MPI job launches in mpi4py comms mode, as it does not support nested MPI launches.
  (Either use local mode or Balsam job controller).
* Local comms mode (multiprocessing) may fail if MPI is initialized before forking processors. This is thought to be responsible for issues combining with PETSc.
* Remote detection of logical cores via LSB_HOSTS (e.g. Summit) returns number of physical cores as SMT info not available.
* TCP mode does not support the auto-resources option to the job controller.


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
