"""
Runs libEnsemble run-lines for adaptive workers in non-persistent case.

Default setup is designed to run on 4*N workers - to modify, change total_nodes.

Execute via one of the following commands (e.g. 8 workers):
   mpiexec -np 9 python test_runlines_adaptive_workers.py

This is a dry run test, mocking up the nodes available. To test the run-lines
requires running a fixed, rather than random number of resource sets for a given sim_id.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 5

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import helloworld
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_with_variable_resources as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample_with_var_priorities_and_resources as gen_f
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.tests.regression_tests.common import create_node_file

nworkers, is_manager, libE_specs, _ = parse_args()

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    # For varying size test - relate node count to nworkers
    total_nodes = nworkers // 4  # 4 workers per node - run on 4N workers

    sim_app = helloworld.__file__
    exctr = MPIExecutor()
    exctr.register_app(full_path=sim_app, app_name="helloworld")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        # 'in': ['x', 'resource_sets'], # Dont need if letting it just use all resources available
        "in": ["x"],
        "out": [("f", float)],
        "user": {"dry_run": True},
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("priority", float), ("resource_sets", int), ("x", float, n), ("x_on_cube", float, n)],
        "user": {
            "initial_batch_size": 5,
            "max_resource_sets": 4,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {
        "alloc_f": give_sim_work_first,
        "user": {
            "batch_mode": False,
            "give_all_with_same_priority": True,
            "num_active_gens": 1,
        },
    }

    comms = libE_specs["comms"]
    node_file = "nodelist_adaptive_workers_comms_" + str(comms) + "_wrks_" + str(nworkers)
    if is_manager:
        create_node_file(num_nodes=total_nodes, name=node_file)

    if comms == "mpi":
        libE_specs["mpi_comm"].Barrier()

    # Mock up system
    libE_specs["resource_info"] = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,
    }  # Name of file containing a node-list

    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"sim_max": 40, "wallclock_max": 300}

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
    )

    if is_manager:
        assert flag == 0
        save_libE_output(H, persis_info, __file__, nworkers)
