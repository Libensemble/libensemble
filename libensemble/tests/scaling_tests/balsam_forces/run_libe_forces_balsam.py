#!/usr/bin/env python
import numpy as np

from libensemble import Ensemble
from libensemble.executors import NewBalsamExecutor
from balsam.api import ApplicationDefinition

# Use Globus to transfer output forces.stat files back
TRANSFER_STATFILES = True
GLOBUS_ENDPOINT = "jln_laptop"

forces = Ensemble()
forces.from_yaml("balsam_forces.yaml")

forces.gen_specs["user"].update({"lb": np.array([0]), "ub": np.array([32767])})
forces.sim_specs["user"].update({"transfer": TRANSFER_STATFILES, "globus_endpoint": GLOBUS_ENDPOINT})

forces.persis_info.add_random_streams()


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


exctr = NewBalsamExecutor()
exctr.register_app(RemoteForces, app_name="forces")

batch = exctr.submit_allocation(
    site_id=246,
    num_nodes=4,
    wall_time_min=30,
    queue="debug-flat-quad",
    project="CSC250STMS07",
)

forces.run()

exctr.revoke_allocation(batch)
