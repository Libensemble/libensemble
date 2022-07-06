#!/usr/bin/env python
import os
import socket
import numpy as np

from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample
from forces_simf import run_forces_balsam
from libensemble.executors.balsam_executors import BalsamExecutor
from libensemble.tools import parse_args, add_unique_random_streams

from balsam.api import ApplicationDefinition

BALSAM_SITE = "three"

# Is this running on a personal machine, or a compute node?
THIS_SCRIPT_ON_THETA = any([i in socket.gethostname() for i in ["theta", "nid0"]])

# Use Globus to transfer output forces.stat files back
GLOBUS_ENDPOINT = "jln_laptop"

if not THIS_SCRIPT_ON_THETA:
    GLOBUS_DEST_DIR_PREFIX = os.getcwd() + "/ensemble"
else:
    GLOBUS_DEST_DIR_PREFIX = "/path/to/remote/ensemble/directory"

# Parse number of workers, comms type, etc. from arguments
nworkers, is_manager, libE_specs, _ = parse_args()

# State the sim_f, inputs, outputs
sim_specs = {
    "sim_f": run_forces_balsam,  # sim_f, imported above
    "in": ["x"],  # Name of input for sim_f
    "out": [("energy", float)],  # Name, type of output from sim_f
    "user": {
        "globus_endpoint": GLOBUS_ENDPOINT,
        "globus_dest_dir": GLOBUS_DEST_DIR_PREFIX,
        "this_script_on_theta": THIS_SCRIPT_ON_THETA,
    },
}

# State the gen_f, inputs, outputs, additional parameters
gen_specs = {
    "gen_f": uniform_random_sample,  # Generator function
    "out": [("x", float, (1,))],  # Name, type and size of data from gen_f
    "user": {
        "lb": np.array([1000]),  # User parameters for the gen_f
        "ub": np.array([3000]),
        "gen_batch_size": 8,
    },
}

# Create and work inside separate per-simulation directories
libE_specs["sim_dirs_make"] = True

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": 8}

persis_info = add_unique_random_streams({}, nworkers + 1)

apps = ApplicationDefinition.load_by_site(BALSAM_SITE)
RemoteForces = apps["RemoteForces"]

exctr = BalsamExecutor()
exctr.register_app(RemoteForces, app_name="forces")

if not THIS_SCRIPT_ON_THETA:
    batch = exctr.submit_allocation(
        site_id=246,  # Check if matches BALSAM_SITE with `balsam site ls`
        num_nodes=4,
        wall_time_min=30,
        queue="debug-flat-quad",
        project="CSC250STMS07",
    )

# Launch libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)

if not THIS_SCRIPT_ON_THETA:
    exctr.revoke_allocation(batch)
