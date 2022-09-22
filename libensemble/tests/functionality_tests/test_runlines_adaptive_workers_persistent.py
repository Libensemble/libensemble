"""
Runs libEnsemble run-lines for adaptive workers with persistent gen.

Default setup is designed to run on 4*N + 1 workers - to modify, change total_nodes.
where one worker is a zero-resource persistent gen.

Execute via one of the following commands (e.g. 9 workers):
   mpiexec -np 10 python test_runlines_adaptive_workers_persistent.py

This is a dry run test, mocking up the nodes available. To test the run-lines
requires running a fixed, rather than random number of resource sets for a given sim_id.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 10

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import helloworld
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_with_variable_resources as sim_f

from libensemble.gen_funcs.persistent_sampling import uniform_random_sample_with_variable_resources as gen_f

# from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.tests.regression_tests.common import create_node_file

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    num_gens = 1
    libE_specs["num_resource_sets"] = nworkers - num_gens  # Any worker can be the gen

    total_nodes = (nworkers - num_gens) // 4  # 4 resourced workers per node.

    if total_nodes == 1:
        max_rsets = 4  # Up to one node
    else:
        max_rsets = 6  # Will expand to 2 full nodes

    sim_app = helloworld.__file__
    exctr = MPIExecutor()
    exctr.register_app(full_path=sim_app, app_name="helloworld")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"dry_run": True},
    }

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["x", "f", "sim_id"],
        "out": [("priority", float), ("resource_sets", int), ("x", float, n), ("x_on_cube", float, n)],
        "user": {
            "initial_batch_size": nworkers - 1,
            "max_resource_sets": max_rsets,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {"give_all_with_same_priority": False},
    }

    comms = libE_specs["comms"]
    node_file = "nodelist_adaptive_workers_persistent_comms_" + str(comms) + "_wrks_" + str(nworkers)
    if is_manager:
        create_node_file(num_nodes=total_nodes, name=node_file)

    if comms == "mpi":
        libE_specs["mpi_comm"].Barrier()

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
