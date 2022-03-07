import time
import glob
from balsam.api import ApplicationDefinition, BatchJob

# Batch Session Parameters
SIM_MAX = 16  # make sure matches in balsam_forces.yaml
BATCH_NUM_NODES = 5
BATCH_WALL_CLOCK_TIME = 60
PROJECT = "CSC250STMS07"
QUEUE = "debug-flat-quad"

# libE Job Parameters - Will use above resources
LIBE_NODES = 1
LIBE_RANKS = 5

# Transfer forces.stat files back to this script's source directory?
#  Adjust run_libe_forces_balsam.py as well!!!!
TRANSFER_STATFILES = True

# Transfer this file to the libE Job's working directory.
#  # globus_endpoint_key *specified in local balsam site's settings.yml*
#  globus_endpoint_key:/path/to/file
input_file = (
    "jln_laptop:/Users/jnavarro/Desktop/libensemble"
    + "/libensemble/libensemble/tests/scaling_tests/balsam_forces/balsam_forces.yaml"
)

# FOR EACH OF THE FOLLOWING APPS, make sure Balsam sites, home directories,
#  pythons, and other paths are updated.


class LibensembleApp(ApplicationDefinition):
    site = "jln_theta"
    command_template = (
        "/home/jnavarro/.conda/envs/again/bin/python /home/jnavarro"
        + "/libensemble/libensemble/tests/scaling_tests/balsam_forces/run_libe_forces_balsam.py"
        + " > libe_out.txt 2>&1"
    )

    transfers = {
        "input_file": {
            "required": True,
            "direction": "in",
            "local_path": ".",
            "description": "Transfer in of balsam_forces.yaml",
            "recursive": False,
        }
    }


print("Defined LibensembleApp Balsam ApplicationDefinition.")

libe_job = LibensembleApp.submit(
    workdir="libe_workflow/libe_processes",
    num_nodes=LIBE_NODES,
    ranks_per_node=LIBE_RANKS,
    transfers={"input_file": input_file},
)

print("libEnsemble Job created, synced with Balsam. Will run on next BatchJob")


class RemoteForces(ApplicationDefinition):
    site = "jln_theta"
    command_template = (
        "/home/jnavarro"
        + "/libensemble/libensemble/tests/scaling_tests/forces/forces.x"
        + " {{sim_particles}} {{sim_timesteps}} {{seed}} {{kill_rate}}"
        + " > out.txt 2>&1"
    )

    transfers = {
        "result": {
            "required": False,
            "direction": "out",
            "local_path": "forces.stat",
            "description": "Forces stat file",
            "recursive": False,
        }
    }


RemoteForces.sync()

print("Defined and synced RemoteForces Balsam ApplicationDefinition.")

batch = BatchJob.objects.create(
    site_id=libe_job.site_id,
    num_nodes=BATCH_NUM_NODES,
    wall_time_min=BATCH_WALL_CLOCK_TIME,
    job_mode="mpi",
    project=PROJECT,
    queue=QUEUE,
)

print("BatchJob session initialized. All Balsam apps will run in this BatchJob.")

if TRANSFER_STATFILES:
    print("Waiting for all returned forces.stat files...")

    while len(glob.glob("./*.stat")) != SIM_MAX:
        time.sleep(3)

    print("All forces.stat files returned. Cancelling BatchJob session.")

    batch.state = "pending_deletion"
    batch.save()

    print("BatchJob session cancelled. Success!")
