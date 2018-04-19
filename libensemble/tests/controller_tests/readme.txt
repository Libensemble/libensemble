This is a temporary location for testing job_controller interface
Note: libEnsemble must be installed
Should work with Balsam and standalone

Install libEnsemble in project root directory (where setup.py is) with either:

pip install .

or for developer build (if you want to make changes):

pip install -e .


For running on laptop/cluster
-----------------------------

Launch job with python E.g:

> python test_jobcontroller.py


For running with Balsam
-----------------------

1. You must install Balsam (https://balsam.alcf.anl.gov/)

2. For each time you start a new session: In this directory (controller_tests) initialise database and start the database server by sourcing script set.balsam.database.sh. You can modify the database location in the script.

> . set.balsam.database.sh

3. Check in test script the USE_BALSAM = True

4. To reset jobs before each run. Create a test app/job by running create_balsam_job.py for desired test E.g:

> ./create_balsam_job.py test_jobcontroller.py

You can see the created app/job by running:

> balsam ls apps

and:

> balsam ls jobs

5. Run jobs using the balsam launcher E.g:

> balsam launcher --consume-all


