#!/usr/bin/env python
import os
import sys
import numpy as np

from libensemble import Ensemble
from libensemble.executors.mpi_executor import MPIExecutor

####################

sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")

if not os.path.isfile(sim_app):
    sys.exit("forces.x not found - please build first in ../forces_app dir")


####################

forces = Ensemble()
forces.from_yaml("forces.yaml")

forces.logger.set_level("INFO")

if forces.is_manager:
    print(f"\nRunning with {forces.nworkers} workers\n")

exctr = MPIExecutor()
exctr.register_app(full_path=sim_app, app_name="forces")

forces.libE_specs["ensemble_dir_path"] = "./ensemble"
forces.gen_specs["user"].update(
    {
        "lb": np.array([0]),
        "ub": np.array([32767]),
    }
)

forces.persis_info.add_random_streams()

forces.run()

if forces.is_manager:
    forces.save_output(__file__)
