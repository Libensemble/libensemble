#!/usr/bin/env python
import numpy as np

from libensemble import Ensemble
from libensemble.executors import NewBalsamExecutor
from balsam.api import ApplicationDefinition

# Use Globus to transfer output forces.stat files back
ON_THETA = True
TRANSFER_STATFILES = False
GLOBUS_ENDPOINT = "jln_laptop"

forces = Ensemble()
forces.from_yaml("balsam_forces.yaml")

forces.gen_specs["user"].update({"lb": np.array([0]), "ub": np.array([32767])})
forces.sim_specs["user"].update({"transfer": TRANSFER_STATFILES, "globus_endpoint": GLOBUS_ENDPOINT})

forces.persis_info.add_random_streams()

apps = ApplicationDefinition.load_by_site("jln_theta")
RemoteForces = apps["RemoteForces"]

exctr = NewBalsamExecutor()
exctr.register_app(RemoteForces, app_name="forces")

if not ON_THETA:
    batch = exctr.submit_allocation(
        site_id=246,
        num_nodes=4,
        wall_time_min=30,
        queue="debug-flat-quad",
        project="CSC250STMS07",
    )

forces.run()

if not ON_THETA:
    exctr.revoke_allocation(batch)
