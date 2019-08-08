import subprocess
import os
import balsam

# Balsam is meant for HPC systems that commonly distribute jobs across many
#   nodes. Due to the nature of testing Balsam on local or CI systems which usually
#   only contain a single node, we need to change Balsam's default worker setup
#   so multiple workers can be run on a single node (until this feature is [hopefully] added!).
#   For our purposes, we append ten workers to Balsam's WorkerGroup


### MODIFY BALSAM WORKERGROUP ###

home = os.getcwd()
balsam_worker_path = os.path.dirname(balsam.__file__) + '/launcher'
os.chdir(balsam_worker_path)

with open('worker.py', 'r') as f:
    lines = f.readlines()

if lines[-3] != "        for idx in range(10):\n":
    lines = lines[:-2] # Will re-add these lines
    lines.extend(["        for idx in range(10):\n",
                  "            w = Worker(1, host_type='DEFAULT', num_nodes=1)\n",
                  "            self.workers.append(w)\n"])

with open('testworker.py', 'w') as f:
    for line in lines:
        f.write(line)

os.chdir(home)

### EXECUTE BALSAM JOB ###

runstr = 'balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1'
subprocess.check_call(runstr.split(' '))
