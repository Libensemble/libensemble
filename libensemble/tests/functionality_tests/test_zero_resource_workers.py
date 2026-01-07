"""
Runs libEnsemble testing the zero_resource_workers argument.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_zero_resource_workers.py
   python test_zero_resource_workers.py --nworkers 3
   python test_zero_resource_workers.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import sys

import numpy as np

from libensemble import logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
from libensemble.libE import libE
from libensemble.sim_funcs.run_line_check import runline_check as sim_f
from libensemble.tests.regression_tests.common import create_node_file
from libensemble.tools import add_unique_random_streams, parse_args

# logger.set_level("DEBUG")  # For testing the test
logger.set_level("INFO")

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 4

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    libE_specs["zero_resource_workers"] = [1]
    libE_specs["dedicated_mode"] = True
    libE_specs["enforce_worker_core_bounds"] = True

    # To allow visual checking - log file not used in test
    log_file = "ensemble_zrw_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    nodes_per_worker = 2

    # For varying size test - relate node count to nworkers
    in_place = libE_specs["zero_resource_workers"]
    n_gens = len(in_place)
    nsim_workers = nworkers - n_gens

    comms = libE_specs["comms"]
    nodes_per_worker = 2
    node_file = "nodelist_zero_resource_workers_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = nsim_workers * nodes_per_worker

    # Mock up system
    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,
    }  # Name of file containing a node-list
    libE_specs["resource_info"] = custom_resources

    if is_manager:
        create_node_file(num_nodes=nnodes, name=node_file)

    if comms == "mpi":
        libE_specs["mpi_comm"].Barrier()

    # Mock up system
    mpi_customizer = {
        "mpi_runner": "mpich",  # Select runner: mpich, openmpi, aprun, srun, jsrun
        "runner_name": "mpirun",
    }  # Runner name: Replaces run command if not None

    # Create executor and register sim to it.
    exctr = MPIExecutor(custom_info=mpi_customizer)
    exctr.register_app(full_path=sim_app, calc_type="sim")

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": [],
        "out": [("x", float, (n,))],
        "user": {
            "initial_batch_size": 20,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}
    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"sim_max": (nsim_workers) * rounds}

    # Each worker has 2 nodes. Basic test list for portable options
    test_list_base = [
        {"testid": "base1", "nprocs": 2, "nnodes": 1, "ppn": 2, "e_args": "--xarg 1"},  # Under use
        {"testid": "base2"},  # Give no config and no extra_args
    ]

    exp_mpich = [
        "mpirun -hosts node-1 -np 2 --ppn 2 --xarg 1 /path/to/fakeapp.x --testid base1",
        "mpirun -hosts node-1,node-2 -np 32 --ppn 16 /path/to/fakeapp.x --testid base2",
    ]

    test_list = test_list_base
    exp_list = exp_mpich
    sim_specs["user"] = {
        "tests": test_list,
        "expect": exp_list,
        "nodes_per_worker": nodes_per_worker,
        "persis_gens": n_gens,
    }

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    # All asserts are in sim func
