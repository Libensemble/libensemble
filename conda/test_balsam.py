import subprocess
import os
import time
import sys
from libensemble.tests.regression_tests.common import parse_args, modify_Balsam_worker

# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 3

# This test is NOT submitted as a job to Balsam. That would be script_test_balsam.py
#   This test executes that job through the 'runstr' line defined further down.

nworkers, is_master, libE_specs, _ = parse_args()  # None used. Bug-prevention

# Balsam is meant for HPC systems that commonly distribute jobs across many
#   nodes. Due to the nature of testing Balsam on local or CI systems which usually
#   only contain a single node, we need to change Balsam's default worker setup
#   so multiple workers can be run on a single node.
#   For our purposes, we append ten workers to Balsam's WorkerGroup
print("Currently in {}. Beginning Balsam worker modification".format(os.getcwd()))
modify_Balsam_worker()

# Executes Balsam Job
# By this point, script_test_balsam.py has been submitted as an app and job to Balsam
# This line launches the queued job in the Balsam database
runstr = 'balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1'
print('Executing Balsam job with command: {}'.format(runstr))
subprocess.Popen(runstr.split())

# Location of Balsam DB location defined in configure-balsam-test.sh
basedb = os.path.expanduser('~/test-balsam/data/libe_test-balsam')

# Periodically wait for Workflow and Job directory within Balsam DB
sleeptime = 0
print('{}: Waiting for Workflow directory'.format(sleeptime))
while not os.path.isdir(basedb) and sleeptime < 58:
    sleeptime += 1
    print('{}'.format(sleeptime), end=" ")
    time.sleep(1)

print('{}: Waiting for Job Directory'.format(sleeptime))
while len(os.listdir(basedb)) == 0 and sleeptime < 58:
    sleeptime += 1
    print('{}'.format(sleeptime), end=" ")
    time.sleep(1)

# Periodically check for Balsam general output
jobdirname = os.listdir(basedb)[0]
jobdir = os.path.join(basedb, jobdirname)
outscript = os.path.join(jobdir, 'job_script_test_balsam.out')

# Periodically wait for Balsam Job output
print('{}: Beginning cycle of checking for Balsam output: {}'.format(sleeptime, outscript))
while not os.path.isfile(outscript) and sleeptime < 58:
    sleeptime += 2
    print('{}'.format(sleeptime), end=" ")
    time.sleep(2)

# Print sections of Balsam output to screen every second until complete
print('{}: Balsam job output found! Printing to screen.'.format(sleeptime))
lastposition = 0
lastlines = ['Job 4 done on worker 1\n', 'Job 4 done on worker 2\n']
while sleeptime < 58:
    with open(outscript, 'r') as f:
        f.seek(lastposition)    # Prevents outputting already printed sections
        new = f.read()
        lastposition = f.tell()
    if len(new) > 0:
        print(new)
        sys.stdout.flush()
    if new[-len(lastlines[0]):] in lastlines:
        break
    time.sleep(1)
    sleeptime += 1


print('Test completed.')
