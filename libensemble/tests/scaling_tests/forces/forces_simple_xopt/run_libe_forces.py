#!/usr/bin/env python
import os
import sys

import numpy as np
from forces_simf import run_forces  # Classic libEnsemble sim_f.
from gest_api.vocs import VOCS
from xopt.generators.random import RandomGenerator

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

# from forces_simf import run_forces_dict  # gest-api/xopt style simulator.



if __name__ == "__main__":
    # Initialize MPI Executor
    exctr = MPIExecutor()

    # Register simulation executable with executor
    sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")

    if not os.path.isfile(sim_app):
        sys.exit("forces.x not found - please build first in ../forces_app dir")

    exctr.register_app(full_path=sim_app, app_name="forces")

    # Parse number of workers, comms type, etc. from arguments
    ensemble = Ensemble(parse_args=True, executor=exctr)

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,
        sim_dirs_make=True,
    )

    # Define VOCS specification
    vocs = VOCS(
        variables={"x": [1000, 3000]},  # min and max particles
        objectives={"energy": "MINIMIZE"},
    )

    # Create xopt random sampling generator
    gen = RandomGenerator(vocs=vocs)

    ensemble.gen_specs = GenSpecs(
        initial_batch_size=ensemble.nworkers,
        generator=gen,
        vocs=vocs,
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_forces,
        # simulator=run_forces_dict,
        vocs=vocs,
    )

    # Starts one persistent generator. Simulated values are returned in batch.
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "async_return": False,  # False causes batch returns
        },
    )

    # Instruct libEnsemble to exit after this many simulations
    ensemble.exit_criteria = ExitCriteria(sim_max=8)

    # Run ensemble
    ensemble.run()

    if ensemble.is_manager:
        # Note, this will change if changing sim_max, nworkers, lb, ub, etc.
        print(f'Final energy checksum: {np.sum(ensemble.H["energy"])}')
