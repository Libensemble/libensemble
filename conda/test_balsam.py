import subprocess
import os
import time
from libensemble.tests.regression_tests.common import parse_args, modify_Balsam_worker

# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 3

nworkers, is_master, libE_specs, _ = parse_args()  # None used. Bug-prevention

print('Sleeping for a couple seconds (just in case)')
time.sleep(5)

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
try:
    subprocess.check_output(runstr.split())
except subprocess.CalledProcessError as e:
    print(e.output)

curdir = os.getcwd()

os.chdir('~/test-balsam/data/libe_test-balsam/job_script_test_balsam_*')
with open('job_script_test_balsam.out', 'r') as f:
    lines = f.readlines()
for line in lines:
    print(line)

os.chdir(curdir)
