import subprocess
import os
import balsam
from libensemble.tests.regression_tests.common import parse_args

# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 3

new_lines = ["        for idx in range(10):\n",
             "            w = Worker(1, host_type='DEFAULT', num_nodes=1)\n",
             "            self.workers.append(w)\n"]

nworkers, is_master, libE_specs, _ = parse_args()  # None used. Bug-prevention


# Balsam is meant for HPC systems that commonly distribute jobs across many
#   nodes. Due to the nature of testing Balsam on local or CI systems which usually
#   only contain a single node, we need to change Balsam's default worker setup
#   so multiple workers can be run on a single node (until this feature is [hopefully] added!).
#   For our purposes, we append ten workers to Balsam's WorkerGroup
print("Currently in {}. Beginning Balsam worker modification".format(os.getcwd()))
workerfile = 'worker.py'
home = os.getcwd()
balsam_worker_path = os.path.dirname(balsam.__file__) + '/launcher'
os.chdir(balsam_worker_path)

with open(workerfile, 'r') as f:
    lines = f.readlines()

if lines[-3] != new_lines[0]:
    lines = lines[:-2]  # effectively inserting new_lines[0] above
    lines.extend(new_lines)

with open(workerfile, 'w') as f:
    for line in lines:
        f.write(line)

print("Modified worker file in {}".format(os.getcwd()))
os.chdir(home)


# Executes Balsam Job
# By this point, script_test_balsam.py has been submitted as an app and job to Balsam
# This line launches the queued job in the Balsam database
runstr = 'balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1'
print('Executing Balsam job with command: {}'.format(runstr))
subprocess.call(runstr.split())
