"""
Runs libEnsemble testing the MPI Runners command creation with 2 workers per node.

This test must be run on an even number of workers >= 2 (e.g. odd no. of procs when using mpi4py).

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_mpi_runners_subnode.py
   python test_mpi_runners_subnode.py --nworkers 4
   python test_mpi_runners_subnode.py --nworkers 4 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import sys

import numpy as np

from libensemble import logger
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.libE import libE
from libensemble.sim_funcs.run_line_check import runline_check as sim_f
from libensemble.tests.regression_tests.common import create_node_file
from libensemble.tools import add_unique_random_streams, parse_args

# logger.set_level("DEBUG")  # For testing the test
logger.set_level("INFO")

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    libE_specs["dedicated_mode"] = True
    libE_specs["enforce_worker_core_bounds"] = True

    # To allow visual checking - log file not used in test
    log_file = "ensemble_mpi_runners_subnode_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    nodes_per_worker = 0.5

    # For varying size test - relate node count to nworkers
    nsim_workers = nworkers

    if not (nsim_workers * nodes_per_worker).is_integer():
        sys.exit(f"Sim workers ({nsim_workers}) must divide evenly into nodes")

    comms = libE_specs["comms"]
    node_file = "nodelist_mpi_runners_subnode_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = int(nsim_workers * nodes_per_worker)

    # Mock up system
    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,  # Name of file containing a node-list
    }
    libE_specs["resource_info"] = custom_resources

    if is_manager:
        create_node_file(num_nodes=nnodes, name=node_file)

    if comms == "mpi":
        libE_specs["mpi_comm"].Barrier()

    # Create executor and register sim to it.
    exctr = MPIExecutor(custom_info={"mpi_runner": "srun"})
    exctr.register_app(full_path=sim_app, calc_type="sim")

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
            "gen_batch_size": 20,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"sim_max": (nsim_workers) * rounds}

    # Each worker has 2 nodes. Basic test list for portable options
    test_list_base = [
        {"testid": "base1"},  # Give no config and no extra_args
        {"testid": "base2", "nprocs": 5},
        {"testid": "base3", "nnodes": 1},
        {"testid": "base4", "ppn": 6},
    ]

    exp_srun = [
        "srun -w node-1 --ntasks 8 --nodes 1 --ntasks-per-node 8 --exact /path/to/fakeapp.x --testid base1",
        "srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 --exact /path/to/fakeapp.x --testid base2",
        "srun -w node-1 --ntasks 8 --nodes 1 --ntasks-per-node 8 --exact /path/to/fakeapp.x --testid base3",
        "srun -w node-1 --ntasks 6 --nodes 1 --ntasks-per-node 6 --exact /path/to/fakeapp.x --testid base4",
    ]

    test_list = test_list_base
    exp_list = exp_srun
    sim_specs["user"] = {
        "tests": test_list,
        "expect": exp_list,
        "nodes_per_worker": nodes_per_worker,
    }

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    # All asserts are in sim func
