import os
import time
import glob
from balsam.api import ApplicationDefinition, BatchJob

"""
This file is roughly equivalent to a traditional batch submission shell script
that used legacy Balsam commands, except it uses the Balsam API to submit jobs
to the scheduler. It can also be run from anywhere and still submit jobs to
the same machine. It loads, parameterizes, and submits the LibensembleApp for
execution. Use this script to run libEnsemble as a Balsam Job on compute nodes.

If running libEnsemble on a laptop, this script is not needed. Just run the
corresponding libEnsemble calling script as normal.
"""

BALSAM_SITE = "jln_theta"

# Batch Session Parameters
BATCH_NUM_NODES = 5
BATCH_WALL_CLOCK_TIME = 60
PROJECT = "CSC250STMS07"
QUEUE = "debug-flat-quad"

# libEnsemble Job Parameters - A subset of above resources dedicated to libEnsemble
LIBE_NODES = 1
LIBE_RANKS = 5

# This script cancels remote allocation once SIM_MAX statfiles transferred
TRANSFER_DESTINATION = "./ensemble"
SIM_MAX = 16

# Retrieve the libEnsemble app from the Balsam service
apps = ApplicationDefinition.load_by_site(BALSAM_SITE)
RemoteLibensembleApp = apps["RemoteLibensembleApp"]
RemoteLibensembleApp.resolve_site_id()


# Submit the libEnsemble app as a Job to the Balsam service.
#  It will wait for a compatible, running BatchJob session (remote allocation)
libe_job = RemoteLibensembleApp.submit(
    workdir="libe_workflow",
    num_nodes=LIBE_NODES,
    ranks_per_node=LIBE_RANKS,
)

print("libEnsemble App retrieved and submitted as Job to Balsam service.")

# Submit an allocation (BatchJob) request to the libEnsemble app's site
batch = BatchJob.objects.create(
    site_id=libe_job.site_id,
    num_nodes=BATCH_NUM_NODES,
    wall_time_min=BATCH_WALL_CLOCK_TIME,
    job_mode="mpi",
    project=PROJECT,
    queue=QUEUE,
)

print("BatchJob session initialized. All Balsam apps will run in this BatchJob.")

# Wait for all forces.stat files to be transferred back, then cancel the BatchJob
os.makedirs(TRANSFER_DESTINATION, exist_ok=True)
print("Waiting for all returned forces.stat files...")

while len(glob.glob(os.path.abspath(TRANSFER_DESTINATION) + "/*.stat")) != SIM_MAX:
    time.sleep(3)

print("All forces.stat files returned. Cancelling BatchJob session.")

batch.state = "pending_deletion"
batch.save()

print("BatchJob session cancelled. Success!")
