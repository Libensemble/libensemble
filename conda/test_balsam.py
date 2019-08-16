import subprocess
import os
import time
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

basedb = os.path.expanduser('~/test-balsam/data/libe_test-balsam')

sleeptime = 0

while not os.path.isdir(basedb) and sleeptime < 58:
    print('{}: Waiting for Workflow directory'.format(sleeptime))
    sleeptime += 1
    time.sleep(1)

while len(os.listdir(basedb)) == 0 and sleeptime < 58:
    print('{}: Waiting for Job Directory'.format(sleeptime))
    sleeptime += 1
    time.sleep(1)

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

fileout = []
lastline = None
while True:
    with open(outscript, 'r') as f:
        line = f.readline()
        if line != lastline:
            lastline = line
            fileout.append(line)
            continue
        if not line and len(fileout) > 20:
            break
        continue

for line in fileout:
    print(line)

print('Test completed.')
