=============
Release Notes
=============


Release 0.3.0
-------------

:Date: September 7, 2018

* Issues with killing jobs have been fixed (#21)
* History is dumped on exception and MPI aborted (#46)
* Added directory of standalone tests (comms, job kills, and nested MPI launches)
* Bugfix to job_controller manager_poll to work with multiple jobs (#62)
* Improved a speeded up unit tested (#68)

:Known issues:
* OpenMPI does not support direct launches (and has not been tested with Balsam).


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
