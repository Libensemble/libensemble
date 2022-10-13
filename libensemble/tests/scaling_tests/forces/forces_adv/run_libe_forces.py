#!/usr/bin/env python
import os
import sys
import numpy as np
from forces_simf import run_forces  # Sim func from current dir

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.manager import ManagerException
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble import logger
from forces_support import test_libe_stats, test_ensemble_dir, check_log_exception

# Note the Balsam option here is now LegacyBalsam - see balsam_forces for latest.
USE_BALSAM = False

PERSIS_GEN = False

if PERSIS_GEN:
    from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
    from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
else:
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
    from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first as alloc_f


logger.set_level("INFO")  # INFO is now default

nworkers, is_manager, libE_specs, _ = parse_args()

sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")

if not os.path.isfile(sim_app):
    sys.exit("forces.x not found - please build first in ../forces_app dir")

if is_manager:
    print(f"\nRunning with {nworkers} workers\n")


# Create executor and register sim to it.
if USE_BALSAM:
    from libensemble.executors.legacy_balsam_executor import LegacyBalsamMPIExecutor

    exctr = LegacyBalsamMPIExecutor()
else:
    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor()
exctr.register_app(full_path=sim_app, app_name="forces")

# Note: Attributes such as kill_rate are to control forces tests, this would not be a typical parameter.

# State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {
    "sim_f": run_forces,  # Function whose output is being minimized
    "in": ["x"],  # Name of input for sim_f
    "out": [("energy", float)],  # Name, type of output from sim_f
    "user": {
        "keys": ["seed"],
        "cores": 2,
        "sim_particles": 1e3,
        "sim_timesteps": 5,
        "sim_kill_minutes": 10.0,
        "particle_variance": 0.2,
        "kill_rate": 0.5,
        "fail_on_sim": False,
        "fail_on_submit": False,  # Won't occur if 'fail_on_sim' True
    },
}
# end_sim_specs_rst_tag

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {
    "gen_f": gen_f,  # Generator function
    "in": [],  # Generator input
    "out": [("x", float, (1,))],  # Name, type and size of data produced (must match sim_specs 'in')
    "user": {
        "lb": np.array([0]),  # Lower bound for random sample array (1D)
        "ub": np.array([32767]),  # Upper bound for random sample array (1D)
        "gen_batch_size": 1000,  # How many random samples to generate in one call
    },
}

if PERSIS_GEN:
    alloc_specs = {"alloc_f": alloc_f}
else:
    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "batch_mode": True,  # If true wait for all sims to process before generate more
            "num_active_gens": 1,  # Only one active generator at a time
        },
    }

libE_specs["save_every_k_gens"] = 1000  # Save every K steps
libE_specs["sim_dirs_make"] = True  # Separate each sim into a separate directory
libE_specs["profile"] = False  # Whether to have libE profile on (default False)

# Maximum number of simulations
sim_max = 8
exit_criteria = {"sim_max": sim_max}

# Create a different random number stream for each worker and the manager
persis_info = {}
persis_info = add_unique_random_streams(persis_info, nworkers + 1)

try:
    H, persis_info, flag = libE(
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info=persis_info,
        alloc_specs=alloc_specs,
        libE_specs=libE_specs,
    )

except ManagerException:
    if is_manager and sim_specs["user"]["fail_on_sim"]:
        check_log_exception()
        test_libe_stats("Exception occurred\n")
else:
    if is_manager:
        save_libE_output(H, persis_info, __file__, nworkers)
        if sim_specs["user"]["fail_on_submit"]:
            test_libe_stats("Task Failed\n")
        test_ensemble_dir(libE_specs, "./ensemble", nworkers, sim_max)
