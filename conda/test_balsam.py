import subprocess
import os
import time
import sys
import libensemble
from libensemble.tests.regression_tests.common import parse_args, modify_Balsam_worker

# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 3

# This test is NOT submitted as a job to Balsam. That would be script_test_balsam.py
#   This test executes that job through the 'runstr' line defined further down.

nworkers, is_master, libE_specs, _ = parse_args()  # None used. Bug-prevention

# Set libensemble base directory to environment variable for Balsam coverage
#   orientation purposes. Otherwise coverage in the Balsam data directory is
#   (unsuccessfully) collected.
libepath = os.path.dirname(libensemble.__file__)
os.environ['LIBE_PATH'] = libepath

# Balsam is meant for HPC systems that commonly distribute jobs across many
#   nodes. For our purposes, we append (hack) ten workers to Balsam's WorkerGroup
print("Currently in {}. Beginning Balsam worker modification".format(os.getcwd()))
modify_Balsam_worker()

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
    sys.stdout.flush()
    time.sleep(1)

print('{}: Waiting for Job Directory'.format(sleeptime))
while len(os.listdir(basedb)) == 0 and sleeptime < 58:
    sleeptime += 1
    print('{}'.format(sleeptime), end=" ")
    sys.stdout.flush()
    time.sleep(1)

# Job directory now exists
jobdirname = os.listdir(basedb)[0]
jobdir = os.path.join(basedb, jobdirname)
outscript = os.path.join(jobdir, 'job_script_test_balsam.out')

# Periodically wait for Balsam Job output
print('{}: Checking for Balsam output file: {}'.format(sleeptime, outscript))
while not os.path.isfile(outscript) and sleeptime < 58:
    sleeptime += 2
    print('{}'.format(sleeptime), end=" ")
    sys.stdout.flush()
    time.sleep(2)

# Print sections of Balsam output to screen every second until complete
print('{}: Output file exists. Waiting for Balsam Job Output.'.format(sleeptime))
lastposition = 0
lastlines = ['Job 4 done on worker 1\n', 'Job 4 done on worker 2\n']
while sleeptime < 58:
    with open(outscript, 'r') as f:
        f.seek(lastposition)    # (should) prevent outputting already printed sections
        new = f.read()
        lastposition = f.tell()
    if len(new) > 0:
        print(new)
        sys.stdout.flush()
    if any(new.endswith(line) for line in lastlines):
        break
    print('{}'.format(sleeptime), end=" ")
    time.sleep(1)
    sleeptime += 1

print('{}: Importing any coverage from Balsam.'.format(sleeptime))
here = os.getcwd()

# Move coverage files from Balsam DB to ./regression_tests (for concatenation)
covname = '.cov_reg_out.'
for file in os.listdir(jobdir):
    if file.startswith(covname):
        balsam_cov = os.path.join(jobdir, file)
        here_cov = os.path.join(here, file)
        print('Moved {} from {} to {}'.format(file, jobdir, here))
        os.rename(balsam_cov, here_cov)

print('Test complete.')
