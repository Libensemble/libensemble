import subprocess
import os
import time
import sys
from libensemble.tests.regression_tests.common import parse_args, modify_Balsam_worker

# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 3

nworkers, is_master, libE_specs, _ = parse_args()  # None used. Bug-prevention

# Balsam is meant for HPC systems that commonly distribute jobs across many
#   nodes. Due to the nature of testing Balsam on local or CI systems which usually
#   only contain a single node, we need to change Balsam's default worker setup
#   so multiple workers can be run on a single node (until this feature is [hopefully] added!).
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
while not os.path.isdir(basedb) and sleeptime < 58:
    print('{}: Waiting for Workflow directory'.format(sleeptime))
    sleeptime += 1
    time.sleep(1)

while len(os.listdir(basedb)) == 0 and sleeptime < 58:
    print('{}: Waiting for Job Directory'.format(sleeptime))
    sleeptime += 1
    time.sleep(1)

# Periodically check for Balsam general output
jobdirname = os.listdir(basedb)[0]
jobdir = os.path.join(basedb, jobdirname)
outscript = os.path.join(jobdir, 'job_script_test_balsam.out')

print('Beginning cycle of checking for Balsam output: {}'.format(outscript))
while sleeptime < 58:
    if os.path.isfile(outscript):
        print('{}: Balsam job output found!'.format(sleeptime))
        break
    else:
        print('{}: No job output found. Checking again.'.format(sleeptime))
        sleeptime += 2
        time.sleep(2)

# Output from Balsam Job (script_test_balsam.py)
# print('Writing output file to terminal')
# last = 0
# lastline = 'Job 4 done on worker 1'
# while True:
#     with open(outscript, 'r') as f:
#         f.seek(last)
#         new = f.read()
#         last = f.tell()
#     print(new)
#     sys.stdout.flush()
#     if new[-len(lastline):] == lastline:
#         break


lastline = 'Job 4 done on worker 1\n'
while True:
    with open(outscript, 'r') as f:
        new = f.read()
        time.sleep(0.5)
        if new[-len(lastline):] == lastline:
            break

with open(outscript, 'r') as f:
    lines = f.read()
    print(lines)
    sys.stdout.flush()

print('Test completed.')
