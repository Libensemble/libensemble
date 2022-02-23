#!/usr/bin/env python
import numpy as np

from libensemble import Ensemble
from libensemble.executors import NewBalsamMPIExecutor

from balsam.api import ApplicationDefinition

forces = Ensemble()
forces.from_yaml("balsam_forces.yaml")

forces.gen_specs["user"].update(
    {
        "lb": np.array([0]),
        "ub": np.array([32767]),
    }
)

forces.persis_info.add_random_streams()

class RemoteForces(ApplicationDefinition):
    site = "three"
    command_template = (
        "/Users/jnavarro/Desktop/libensemble/"
        + "libensemble/libensemble/tests/scaling_tests/forces/forces.x"
        + " {{sim_particles}} {{sim_timesteps}} {{seed}} {{kill_rate}}"
        + " > out.txt 2>&1"
    )

    transfers = {
        "result": {
            "required": True,
            "direction": "out",
            "local_path": "forces.stat",
            "description": "Forces stat file",
            "recursive": False
        }
    }

exctr = NewBalsamMPIExecutor()
exctr.register_app(RemoteForces, app_name="forces")

batch = exctr.submit_allocation(
    site_id=239,
    num_nodes=1,
    wall_time_min=30,
)

forces.run()

# exctr.revoke_allocation(batch)
