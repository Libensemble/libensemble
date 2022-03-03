import time
import glob
from balsam.api import ApplicationDefinition, BatchJob

SIM_MAX = 16  # make sure matches in balsam_forces.yaml


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

input_file = (
    "jln_laptop:/Users/jnavarro/Desktop/libensemble"
    + "/libensemble/libensemble/tests/scaling_tests/balsam_forces/balsam_forces.yaml"
)

libe_job = LibensembleApp.submit(
    workdir="libe_workflow/libe_processes",
    num_nodes=1,
    ranks_per_node=5,
    transfers={"input_file": input_file},
)

print("libEnsemble Job created, synced with Balsam.")

batch = BatchJob.objects.create(
    site_id=libe_job.site_id,
    num_nodes=5,
    wall_time_min=60,
    job_mode="mpi",
    project="CSC250STMS07",
    queue="debug-flat-quad",
)

print("BatchJob session initialized.")
print("Waiting for all returned forces.stat files...")

while len(glob.glob("./*.stat")) != SIM_MAX:
    time.sleep(3)

print("All forces.stat files returned. Cancelling BatchJob session.")

batch.state = "pending_deletion"
batch.save()

print("BatchJob session cancelled. Success!")
