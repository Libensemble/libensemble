#!/usr/bin/env python

import os
import sys

import jinja2
import numpy as np
from gx_simf import run_gx  # Sim func from current dir

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


if __name__ == "__main__":

    gx_input_file = "cyclone.in"

    # Initialize MPI Executor
    exctr = MPIExecutor()
    sim_app = os.path.join(os.getcwd(), "/global/cfs/cdirs/m4493/shudson/gx/gx")

    if not os.path.isfile(sim_app):
        sys.exit("gx not found")

    exctr.register_app(full_path=sim_app, app_name="gx")

    # Parse number of workers, comms type, etc. from arguments
    ensemble = Ensemble(parse_args=True, executor=exctr)
    nworkers = ensemble.nworkers

    platform_specs = {"gpu_setting_type": "option_gpus_per_node"}

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,
        sim_dirs_make=True,
        sim_input_dir="/global/cfs/cdirs/m4493/shudson/gx/libe/cyclone_dir",
        platform_specs=platform_specs,
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_gx,
        inputs=["x"],
        outputs=[("f", float)],
        user = {
            "input_filename": gx_input_file,
            "input_names": ["tprim"],
            "plot_heat_flux": True,
        },
    )

    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],  # No input when start persistent generator
        persis_in=["sim_id"],  # Return sim_ids of evaluated points to generator
        outputs=[("x", float, (1,))],
        user={
            "initial_batch_size": nworkers,
            "lb": np.array([2.1]),  # lower bound for input
            "ub": np.array([3.6]),  # upper bound for input
        },
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

    # Seed random streams for each worker, particularly for gen_f
    ensemble.add_random_streams()

    # Run ensemble
    ensemble.run()

    if ensemble.is_manager:
        ensemble.save_output(__file__)
