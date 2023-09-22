#!/usr/bin/env python

"""
This example is similar to the forces_gpu test.

The forces.c application should be built by setting the GPU preprocessor condition
(usually -DGPU) in addition to openMP GPU flags for the given system. See examples
in ../forces_app/build_forces.sh. We recommend running forces.x standalone first
and confirming it is running on the GPU (this is given clearly in the output).

A number of GPUs is requested based on the number of particles (randomly chosen
from the range for each simulation). For simplicity, the number of GPUs requested
is based on a linear split of the range (lb to ub), rather than absolute particle
count.

To mock on a non-GPU system, uncomment the resource_info line in libE_specs. You
will compile forces without the -DGPU option. It is recommended that the ub and/or lb for
particle counts are reduced for CPU performance.
"""

import os
import sys

import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_with_var_gpus as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

if __name__ == "__main__":
    # Initialize MPI Executor
    exctr = MPIExecutor()
    sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")

    if not os.path.isfile(sim_app):
        sys.exit("forces.x not found - please build first in ../forces_app dir")

    exctr.register_app(full_path=sim_app, app_name="forces")

    # Parse number of workers, comms type, etc. from arguments
    ensemble = Ensemble(parse_args=True, executor=exctr)
    nsim_workers = ensemble.nworkers - 1  # One worker is for persistent generator

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=nsim_workers,
        sim_dirs_make=True,
        stats_fmt={"show_resource_sets": True},  # see resource sets in libE_stats.txt
        # resource_info = {"gpus_on_node": 4},  # for mocking GPUs
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_forces,
        inputs=["x"],
        outputs=[("energy", float)],
    )

    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],  # No input when start persistent generator
        persis_in=["sim_id"],  # Return sim_ids of evaluated points to generator
        outputs=[
            ("x", float, (1,)),
            ("num_gpus", int),  # num_gpus auto given to sim when use MPIExecutor.
        ],
        user={
            "initial_batch_size": nsim_workers,
            "lb": np.array([50000]),  # min particles
            "ub": np.array([100000]),  # max particles
            "max_gpus": nsim_workers,
        },
    )

    # Starts one persistent generator. Simulated values are returned in batch.
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "async_return": False,  # False causes batch returns
        },
    )

    # Instruct libEnsemble to exit after this many simulations.
    ensemble.exit_criteria = ExitCriteria(sim_max=8)

    # Seed random streams for each worker, particularly for gen_f
    ensemble.add_random_streams()

    # Run ensemble
    ensemble.run()

    if ensemble.is_manager:
        # Note, this will change if changing sim_max, nworkers, lb, ub, etc.
        if ensemble.exit_criteria.sim_max == 8:
            chksum = np.sum(ensemble.H["energy"])
            assert np.isclose(chksum, 96288744.35136001), f"energy check sum is {chksum}"
            print("Checksum passed")
        else:
            print("Run complete; a checksum has not been provided for the given sim_max")
