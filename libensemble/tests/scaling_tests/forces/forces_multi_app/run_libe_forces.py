#!/usr/bin/env python

"""
This example runs different applications, one that uses only CPUs and one
that uses GPUs. Both use a variable number of processors. The GPU application
uses one GPU per processor. As the generator creates simulations, it randomly
assigns between one and max_proc processors to each simulation, and also randomly
assigns which application is to be run.

The forces.c application should be compiled for the CPU to `forces_cpu.x`, and
for the GPU (setting the GPU preprocessor condition) to `forces_gpu.x`.

For compile lines, see examples in ../forces_app/build_forces.sh.

It is recommended to run this test such that:
    ((nworkers - 1) - gpus_on_node) >= gen_specs["user"][max_procs]

E.g., if running on one node with four GPUs, then use:
    python run_libE_forces.py --nworkers 9

E.g., if running on one node with eight GPUs, then use:
    python run_libE_forces.py --nworkers 17
"""

import os
import sys

import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_diff_simulations as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

if __name__ == "__main__":
    # Initialize MPI Executor instance
    exctr = MPIExecutor()

    # Register simulation executable with executor
    cpu_app = os.path.join(os.getcwd(), "../forces_app/forces_cpu.x")
    gpu_app = os.path.join(os.getcwd(), "../forces_app/forces_gpu.x")

    if not os.path.isfile(cpu_app):
        sys.exit(f"{cpu_app} not found - please build first in ../forces_app dir")
    if not os.path.isfile(gpu_app):
        sys.exit(f"{gpu_app} not found - please build first in ../forces_app dir")

    exctr.register_app(full_path=cpu_app, app_name="cpu_app")
    exctr.register_app(full_path=gpu_app, app_name="gpu_app")

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
        inputs=["x", "app_type"],
        outputs=[("energy", float)],
    )

    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],  # No input when starting persistent generator
        persis_in=["sim_id"],  # Return sim_ids of evaluated points to generator
        outputs=[
            ("x", float, (1,)),
            ("num_procs", int),  # num_procs auto given to sim when using MPIExecutor
            ("num_gpus", int),  # num_gpus auto given to sim when using MPIExecutor
            ("app_type", "S10"),  # select app type (cpu_app or gpu_app)
        ],
        user={
            "initial_batch_size": nsim_workers,
            "lb": np.array([5000]),  # min particles
            "ub": np.array([10000]),  # max particles
            "max_procs": (nsim_workers) // 2,  # Any sim created can req. 1 worker up to max
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
    ensemble.exit_criteria = ExitCriteria(sim_max=nsim_workers * 2)

    # Seed random streams for each worker, particularly for gen_f.
    ensemble.add_random_streams()

    # Run ensemble
    ensemble.run()

    if ensemble.is_manager:
        # Note, this will change if changing sim_max, nworkers, lb, ub, etc.
        chksum = np.sum(ensemble.H["energy"])
        print(f"Final energy checksum: {chksum}")

        exp_chksums = {16: -21935405.696289998, 32: -26563930.6356}
        exp_chksum = exp_chksums.get(ensemble.exit_criteria.sim_max)

        if exp_chksum is not None:
            assert np.isclose(chksum, exp_chksum), f"energy check sum is {chksum}"
            print("Checksum passed")
        else:
            print("Run complete. An expected checksum has not been provided for the given sim_max")
