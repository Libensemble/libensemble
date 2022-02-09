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

if forces.is_manager():
    RemoteForces = ApplicationDefinition.load_by_site("jln_theta").get("RemoteForce")
    if not RemoteForces:

        class RemoteForces(ApplicationDefinition):
            site = "jln_theta"
            command_template = (
                "/home/jnavarro/"
                + "libensemble/libensemble/tests/scaling_tests/balsam_forces/forces.x"
                + " {{sim_particles}} {{sim_timesteps}} {{seed}} {{kill_rate}}"
                + " > out.txt 2>&1"
            )

    exctr = NewBalsamMPIExecutor()
    exctr.register_app(RemoteForces, app_name="forces")
    exctr.submit_allocation(
        site_id=246,
        num_nodes=1,
        wall_time_min=30,
        queue="debug-cache-quad",
        project="CSC250STMS07",
    )

forces.run()
