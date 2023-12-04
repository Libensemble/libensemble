import os
import sys

import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble import Ensemble, logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

logger.set_level("DEBUG")

if __name__ == "__main__":
    # Initialize MPI Executor
    exctr = MPIExecutor(custom_info={"mpi_runner": "openmpi"})

    # Register simulation executable with executor
    sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")

    if not os.path.isfile(sim_app):
        sys.exit("forces.x not found - please build first in ../forces_app dir")

    exctr.register_app(full_path=sim_app, app_name="forces")

    # Parse number of workers, comms type, etc. from arguments
    ensemble = Ensemble(parse_args=True, executor=exctr)

    nsim_workers = ensemble.nworkers - 1  # One worker is for persistent generator

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=nsim_workers, sim_dirs_make=True, ensemble_dir_path="./test_executor_forces_tutorial_2"
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_forces,
        inputs=["x"],
        outputs=[("energy", float)],
    )

    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],  # No input when starting persistent generator
        persis_in=["sim_id"],  # Return sim_ids of evaluated points to generator
        outputs=[("x", float, (1,))],
        user={
            "initial_batch_size": nsim_workers,
            "lb": np.array([1000]),  # min particles
            "ub": np.array([3000]),  # max particles
        },
    )  # gen_specs_end_tag

    # Starts one persistent generator. Simulated values are returned in batch.
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "async_return": True,
        },
    )

    # Instruct libEnsemble to exit after this many simulations
    ensemble.exit_criteria = ExitCriteria(sim_max=8)

    # Seed random streams for each worker, particularly for gen_f
    ensemble.add_random_streams()

    # Run ensemble
    ensemble.run()

    ensemble.save_output(__file__)
