#!/usr/bin/env python
import numpy as np

from libensemble import Ensemble
from libensemble.executors import BalsamExecutor
from balsam.api import ApplicationDefinition

THIS_SCRIPT_ON_THETA = True  # Is this running on a personal machine, or a compute node?

# Use Globus to transfer output forces.stat files back
TRANSFER_STATFILES = True
GLOBUS_ENDPOINT = "jln_laptop"
GLOBUS_DEST_DIR = (
    "/Users/jnavarro/Desktop/libensemble"
    + "/libensemble/libensemble/tests/scaling_tests/balsam_forces"
)

forces = Ensemble()
forces.from_yaml("balsam_forces.yaml")

forces.gen_specs["user"].update({"lb": np.array([0]), "ub": np.array([32767])})
forces.sim_specs["user"].update(
    {
        "transfer": TRANSFER_STATFILES,
        "globus_endpoint": GLOBUS_ENDPOINT,
        "globus_dest_dir": GLOBUS_DEST_DIR,
        "this_script_on_theta": THIS_SCRIPT_ON_THETA,
    }
)

forces.persis_info.add_random_streams()

apps = ApplicationDefinition.load_by_site("jln_theta")
RemoteForces = apps["RemoteForces"]

exctr = BalsamExecutor()
exctr.register_app(RemoteForces, app_name="forces")

if not THIS_SCRIPT_ON_THETA:
    batch = exctr.submit_allocation(
        site_id=246,
        num_nodes=4,
        wall_time_min=30,
        queue="debug-flat-quad",
        project="CSC250STMS07",
    )

forces.run()

if not THIS_SCRIPT_ON_THETA:
    exctr.revoke_allocation(batch)
