"""
Runs libEnsemble run-lines for adaptive workers with persistent gen.

Default setup is designed to run on 2*N + 1 workers - to modify, change total_nodes.
where one worker is a zero-resource persistent gen.

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 6 python test_runlines_adaptive_workers_persistent_oversubscribe_rsets.py

This is a dry run test, mocking up the nodes available. To test the run-lines
requires running a fixed, rather than random number of resource sets for a given sim_id.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 6

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_with_var_priorities as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import helloworld
from libensemble.sim_funcs.var_resources import multi_points_with_variable_resources as sim_f
from libensemble.tests.regression_tests.common import create_node_file
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    nsim_workers = nworkers - 1

    libE_specs["zero_resource_workers"] = [1]
    rsets = nsim_workers * 2
    libE_specs["num_resource_sets"] = rsets

    num_gens = len(libE_specs["zero_resource_workers"])
    total_nodes = (nworkers - num_gens) // 2  # 2 resourced workers per node.

    print(f"sim_workers: {nsim_workers}.  rsets: {rsets}.  Nodes: {total_nodes}", flush=True)

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
        "persis_in": ["f", "x", "sim_id"],
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

    # comms = libE_specs["disable_resource_manager"] = True # SH TCP testing

    comms = libE_specs["comms"]
    node_file = "nodelist_adaptive_workers_persistent_ovsub_rsets_comms_" + str(comms) + "_wrks_" + str(nworkers)
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
