"""
Runs libEnsemble testing the MPI Runners command creation with uneven workers per node.

This test must be run on an odd number of workers >= 3 and <= 31 (e.g. even no. of procs when using mpi4py).

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 6 python test_mpi_runners_subnode_uneven.py
   python test_mpi_runners_subnode_uneven.py --nworkers 5
   python test_mpi_runners_subnode_uneven.py --nworkers 5 --comms tcp
"""

import sys

import numpy as np

from libensemble import logger
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.libE import libE
from libensemble.sim_funcs.run_line_check import runline_check_by_worker as sim_f
from libensemble.tests.regression_tests.common import create_node_file
from libensemble.tools import add_unique_random_streams, parse_args

# logger.set_level("DEBUG")  # For testing the test
logger.set_level("INFO")

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4 6

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    # To allow visual checking - log file not used in test
    log_file = "ensemble_mpi_runners_subnode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    # For varying size test - relate node count to nworkers
    nsim_workers = nworkers

    if nsim_workers % 2 == 0:
        sys.exit(f"This test must be run with an odd of workers >= 3 and <= 31. There are {nsim_workers} workers.")

    comms = libE_specs["comms"]
    node_file = "nodelist_mpi_runners_subnode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = 2

    if is_manager:
        create_node_file(num_nodes=nnodes, name=node_file)

    if comms == "mpi":
        libE_specs["mpi_comm"].Barrier()

    # Mock up system
    mpi_customizer = {
        "mpi_runner": "srun",  # Select runner: mpich, openmpi, aprun, srun, jsrun
        "runner_name": "srun",
    }  # Runner name: Replaces run command if not None

    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,
    }  # Name of file containing a node-list

    libE_specs["dedicated_mode"] = True
    libE_specs["enforce_worker_core_bounds"] = True
    libE_specs["resource_info"] = custom_resources

    # Create executor and register sim to it.
    exctr = MPIExecutor(custom_info=mpi_customizer)
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

    test_list_base = [
        {"testid": "base1"},  # Give no config and no extra_args
    ]

    # Example: On 5 workers, runlines should be ...
    # [w1]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 --exact /path/to/fakeapp.x --testid base1
    # [w2]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 --exact /path/to/fakeapp.x --testid base1
    # [w3]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 --exact /path/to/fakeapp.x --testid base1
    # [w4]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 --exact /path/to/fakeapp.x --testid base1
    # [w5]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 --exact /path/to/fakeapp.x --testid base1

    srun_p1 = "srun -w "
    srun_p2 = " --ntasks "
    srun_p3 = " --nodes 1 --ntasks-per-node "
    srun_p4 = " --exact /path/to/fakeapp.x --testid base1"

    exp_tasks = []
    exp_srun = []

    # Hard coding an example for 2 nodes to avoid replicating general logic in libEnsemble.
    low_wpn = nsim_workers // nnodes
    high_wpn = nsim_workers // nnodes + 1

    for i in range(nsim_workers):
        if i < (nsim_workers // nnodes + 1):
            nodename = "node-1"
            ntasks = 16 // high_wpn
        else:
            nodename = "node-2"
            ntasks = 16 // low_wpn
        exp_tasks.append(ntasks)
        exp_srun.append(srun_p1 + str(nodename) + srun_p2 + str(ntasks) + srun_p3 + str(ntasks) + srun_p4)

    test_list = test_list_base
    exp_list = exp_srun
    sim_specs["user"] = {
        "tests": test_list,
        "expect": exp_list,
        "offset_for_schedular": True,
    }

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    # All asserts are in sim func
