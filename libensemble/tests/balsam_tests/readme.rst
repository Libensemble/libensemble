============
Using Balsam
============

Note: To set up Balsam - see instructions on setting up in Balsam docs.

Currently this can be found in the ``docs/`` subdirectory of the ``hpc-edge-service``
repository. The documentation is in Sphinx format.

Theta:
If the instructions are followed to set up a conda environment called Balsam,
then the script env_setup_theta.sh, under the Balsam tests directory, can be sourced
in future logins for theta. Or modified for other platforms. In some platforms
you may need to only activate the Balsam conda environment.

-----------------------------------------
Quickstart - Balsam tests for libEnsemble
-----------------------------------------

Having set up Balsam, the tests here can be run as follows.

**i) Set up the tests**

Go to ``balsam_tests`` directory run setup script:

.. code-block:: bash

  cd libensemble/code/tests/balsam_tests
  ./setup_balsam_tests.py

This will register the applications and jobs in the database. If work_dir and
sim_input_dir paths are correct, it can be run from anywhere.

**ii) Launch the jobs**

In an interactive session (or a batch script) run the launcher:

.. code-block:: bash

  balsam launcher --consume-all

Note: You will need at least 2 nodes, one for parent job and one for user apps,
if you are on a machine where the scheduler does not split nodes. However, it
is recommended that 5 nodes are used for good concurrency.

**iii) Output and reset**

Output will go into test output directories inside the ``balsam_tests`` directory - they will be
created on first run.

To quickly reset the tests to run again use the reset (python) script:

.. code-block:: bash

  ./reset_balsam_tests.py

The file readme.balsam_tests.txt contains a brief explanation of the tests

-------------
General Usage
-------------

1. Register applications
------------------------

You must register with Balsam the parent application (e.g., libEnsemble) and any
user application (e.g., sim funcs/gen funcs).
This only need be done once - unless, for example, the name of the application changes.

Example (as used in tests) - run from within ``balsam_tests`` directory:

Register tests:

.. code-block:: bash

    balsam app --name test_balsam_1__runjobs     --exec test_balsam_1__runjobs.py     --desc "Run balsam test 1"

    balsam app --name test_balsam_2__workerkill  --exec test_balsam_2__workerkill.py  --desc "Run balsam test 2"

    balsam app --name test_balsam_3__managerkill --exec test_balsam_3__managerkill.py --desc "Run balsam test 3"

Register user application that will be called inside tests:

.. code-block:: bash

    balsam app --name helloworld --exec ../../examples/sim_funcs/helloworld.py  --desc "Run helloworld user app"

Note: The ``--exec arg`` is the location of the script or executable, and can be absolute
or relative path (the ``balsam app`` command will convert it to a full path).

To list apps:

.. code-block:: bash

  balsam ls apps

To clean:

.. code-block:: bash

  balsam rm apps --all

2 Register job/s
----------------

This is the job you intend to run. It will reference an application you have
set up.

For example, set up job for test_balsam_1:

Where WORK_DIR is set to output directory for job.

.. code-block:: bash

  balsam job --name job_test_balsam_1__runjobs
               --workflow libe_workflow
               --application test_balsam_1
               --wall-min 1 --num-nodes 1 --ranks-per-node 4
               --url-out="local:$WORK_DIR" --stage-out-files="job_test_balsam_1__runjobs*"

A working directory is set up when the job is run - by default under the Balsam
space e.g.,: ``hpc-edge-service/data/balsamjobs/`` Under this directory a workflow
directory is created (e.g., libe_workflow in above case). From there, files to
keep are staged out as specified by directory in --url-out (use local: for file
directory). The files to stage out are specified by --stage-out-files.
A log will also be created when run under hpc-edge-service/log/

The standard output will go to file <jobname>.out. So in above case this will
be job_balsam1.out which will be staged out to $WORKDIR

In this case 4 ranks per node and 1 node are selected. This is for running on
the parent application (e.g., libEnsemble). This does not constrain the running
of sub-apps (e.g., helloworld), which will use the full allocation available.

Note that the user jobs (launched by libEnsemble) are registered from
within the code. For staging out files, the output directory needs to somehow
be accessible to the code. For the tests here, this is simply the directory of
the test scripts (accessed via the __file__ variable in python). Search for
dag.add_job in test scripts (e.g., test_balsam_1__runjobs.py)

To list jobs:

.. code-block:: bash

  balsam ls jobs

To clean:

.. code-block:: bash

  balsam rm jobs --all

3 Launch job/s
--------------

In an interactive session (or a batch script) run the launcher:

Launch all jobs:

.. code-block:: bash

  balsam launcher --consume-all

For other launcher options:

.. code-block:: bash

  balsam launcher -h

4 Reset jobs
------------

A script to reset the tests is available: reset_balsam_tests.py

This script can be modified easily. However, to reset from the command line -
without removing and re-adding jobs you can do the following.

Note: After running tests the Balsam job database will contain something like
the following (job_ids abbreviated for space):

.. code-block:: bash

  $ balsam ls jobs

::

     job_id            | name                            | workflow       | application    | latest update
    -------------------------------------------------------------------------------------------------------------
     29add031-8e7c-... | job_balsam1                     | libe_workflow  | test_balsam_1  | [01-30-2018 18:57:47 JOB_FINISHED]
     9ca5f106-3fb5-... | outfile_for_sim_id_0_ranks3.txt | libe_workflow  | helloworld     | [01-30-2018 18:55:18 JOB_FINISHED]
     6a607a91-782c-... | outfile_for_sim_id_0_ranks0.txt | libe_workflow  | helloworld     | [01-30-2018 18:55:31 JOB_FINISHED]
     3638ee63-0ecc-... | outfile_for_sim_id_0_ranks2.txt | libe_workflow  | helloworld     | [01-30-2018 18:55:44 JOB_FINISHED]
     a2f08c72-fc0c-... | outfile_for_sim_id_0_ranks1.txt | libe_workflow  | helloworld     | [01-30-2018 18:55:57 JOB_FINISHED]
     183c5f01-a8df-... | outfile_for_sim_id_1_ranks3.txt | libe_workflow  | helloworld     | [01-30-2018 18:56:10 JOB_FINISHED]
    ..................

To remove only the generated jobs you can just use a sub-string of the job name, for example:

.. code-block:: bash

  balsam rm jobs --name outfile

.. code-block:: bash

  $ balsam ls jobs

::

     job_id            | name             | workflow        | application      | latest update
    -----------------------------------------------------------------------------------------------------------------------
     29add031-8e7c-... | job_balsam1      | libe_workflow   | test_balsam_1    | [01-30-2018 18:57:47 JOB_FINISHED]

To run again - change status attribute to READY (you need to specify job_id -
an abbreviation is OK) For example:

.. code-block:: bash

  balsam modify jobs 29ad --attr state --value READY

Now you are ready to re-run.

Theta tip - Interactive sessions
--------------------------------

Interactive sessions can be launched as:

.. code-block:: bash

  qsub -A <project_code> -n 5 -q debug-flat-quad -t 60 -I

This would be a 60 minute interactive session with 5 nodes. You must have a
project code.

You will need to load the conda environment in the interactive session - or source the
script env_setup_theta.sh.

At time of writing theta does not log you out of interactive sessions. But jobs
launched after time is up will not work.

To see time remaining:

.. code-block:: bash

  qstat -fu <username>
