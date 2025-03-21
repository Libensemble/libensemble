"""
Runs libEnsemble testing the MPI Runners command creation.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_mpi_runners.py
   python test_mpi_runners.py --nworkers 3
   python test_mpi_runners.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

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
# TESTSUITE_NPROCS: 2 4

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    libE_specs["dedicated_mode"] = True
    libE_specs["enforce_worker_core_bounds"] = True

    # To allow visual checking - log file not used in test
    log_file = "ensemble_mpi_runners_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    nodes_per_worker = 2

    # For varying size test - relate node count to nworkers
    node_file = "nodelist_mpi_runners_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = nworkers * nodes_per_worker

    if is_manager:
        create_node_file(num_nodes=nnodes, name=node_file)

    if comms == "mpi":
        libE_specs["mpi_comm"].Barrier()

    # Mock up system
    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,  # Name of file containing a node-list
        "gpus_on_node": 4,  # For tests with ngpus
    }
    libE_specs["resource_info"] = custom_resources

    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"sim_max": nworkers * rounds}

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, (2,))],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "gen_batch_size": 100,
        },
    }

    # Each worker has 2 nodes. Basic test list for portable options
    test_list_base = [
        {"testid": "base1", "nprocs": 2, "nnodes": 1, "ppn": 2, "e_args": "--xarg 1"},  # Under use
        {"testid": "base2"},  # Give no config and no extra_args
        {"testid": "base3", "e_args": "--xarg 1"},  # Give no config with extra_args
        {"testid": "base4", "e_args": "--xarg 1", "ht": True},  # Give no config but with HT
        {"testid": "base5", "nprocs": 16, "e_args": "--xarg 1"},  # Just nprocs (will use one node)
        {"testid": "base6", "nprocs": 16, "nnodes": 2, "e_args": "--xarg 1"},  # nprocs, nnodes
    ]

    # extra_args tests for each mpi runner
    # extra_args should be parsed - however, the string should stay intact, inc abbreviated/different expressions
    eargs_mpich = [
        {"testid": "mp1", "nprocs": 16, "e_args": "--xarg 1 --ppn 16"},  # nprocs + parse extra_args
        {"testid": "mp2", "e_args": "-np 8 --xarg 1 --ppn 4"},  # parse extra_args
    ]

    eargs_openmpi = [
        {"testid": "ompi1", "nprocs": 16, "e_args": "--xarg 1 -npernode 16"},  # nprocs + parse extra_args
        {"testid": "ompi2", "e_args": "-np 8 --xarg 1 -npernode 4"},  # parse extra_args
    ]

    eargs_aprun = [
        {"testid": "ap1", "nprocs": 16, "e_args": "--xarg 1 -N 16"},  # nprocs + parse extra_args
        {"testid": "ap2", "e_args": "-n 8 --xarg 1 -N 4"},  # parse extra_args
    ]

    # Note in a8: -n 8 is abbreviated form of --ntasks, this should be unaltered while --nodes is derived and inserted.
    eargs_srun = [
        {"testid": "sr1", "nprocs": 16, "e_args": "--xarg 1 --ntasks-per-node 16"},  # nprocs + parse extra_args
        {"testid": "sr2", "e_args": "-n 8 --xarg 1 --ntasks-per-node 4"},  # parse extra_args
        {"testid": "sr3", "e_args": "--nodes 2 -n 8 --xarg 1 --ntasks-per-node 4"},
        {"testid": "sr4", "ngpus": 8, "e_args": "--nodes 2 -n 8 --xarg 1 --ntasks-per-node 4"},
        {"testid": "sr5", "ngpus": 8, "e_args": "-n 8 --xarg 1 --ntasks-per-node 4"},
    ]

    # Note for jsrun: proc = resource set. Awkward naming but this seems like the best solution.
    # Define extra_args as minimal relation of tasks/cores/gpus (one resource set), then n (nprocs) as multiplier.
    eargs_jsrun = [
        {"testid": "jsr1", "nprocs": 16, "e_args": "--xarg 1 -r 16"},  # nprocs + parse extra_args
        {"testid": "jsr2", "e_args": "-n 8 --xarg 1 -r 4"},  # parse extra_args
        {"testid": "jsr3", "nprocs": 3, "e_args": '-a 1 -c 1 -g 1 --bind=packed:1 --smpiargs="-gpu"'},  # combine r-sets
        {
            "testid": "jsr4",
            "e_args": '-n 3 -a 1 -c 1 -g 1 --bind=packed:1 --smpiargs="-gpu"',
        },  # r-sets all in extra_args
    ]

    eargs_custom = [{"testid": "cust1", "e_args": "--xarg 1 --ppn 16"}]

    # Expected Outputs
    # Node IDs are modified for different workers in sim func.
    exp_mpich = [
        "mpirun -hosts node-1 -np 2 --ppn 2 --xarg 1 /path/to/fakeapp.x --testid base1",
        "mpirun -hosts node-1,node-2 -np 32 --ppn 16 /path/to/fakeapp.x --testid base2",
        "mpirun -hosts node-1,node-2 -np 32 --ppn 16 --xarg 1 /path/to/fakeapp.x --testid base3",
        "mpirun -hosts node-1,node-2 -np 128 --ppn 64 --xarg 1 /path/to/fakeapp.x --testid base4",
        "mpirun -hosts node-1 -np 16 --ppn 16 --xarg 1 /path/to/fakeapp.x --testid base5",
        "mpirun -hosts node-1,node-2 -np 16 --ppn 8 --xarg 1 /path/to/fakeapp.x --testid base6",
        "mpirun -hosts node-1 -np 16 --xarg 1 --ppn 16 /path/to/fakeapp.x --testid mp1",
        "mpirun -hosts node-1,node-2 -np 8 --xarg 1 --ppn 4 /path/to/fakeapp.x --testid mp2",
    ]

    exp_rename_mpich = [
        "inst -dummy mpich -hosts node-1 -np 2 --ppn 2 --xarg 1 /path/to/fakeapp.x --testid base1",
        "inst -dummy mpich -hosts node-1,node-2 -np 32 --ppn 16 /path/to/fakeapp.x --testid base2",
        "inst -dummy mpich -hosts node-1,node-2 -np 32 --ppn 16 --xarg 1 /path/to/fakeapp.x --testid base3",
        "inst -dummy mpich -hosts node-1,node-2 -np 128 --ppn 64 --xarg 1 /path/to/fakeapp.x --testid base4",
        "inst -dummy mpich -hosts node-1 -np 16 --ppn 16 --xarg 1 /path/to/fakeapp.x --testid base5",
        "inst -dummy mpich -hosts node-1,node-2 -np 16 --ppn 8 --xarg 1 /path/to/fakeapp.x --testid base6",
        "inst -dummy mpich -hosts node-1 -np 16 --xarg 1 --ppn 16 /path/to/fakeapp.x --testid mp1",
        "inst -dummy mpich -hosts node-1,node-2 -np 8 --xarg 1 --ppn 4 /path/to/fakeapp.x --testid mp2",
    ]

    # openmpi requires machinefiles (-host requires
    # exp_openmpi = \
    #    ['mpirun -host node-1 -np 2 -npernode 2 --xarg 1 /path/to/fakeapp.x --testid base1',
    #    'mpirun -host node-1,node-2 -np 32 -npernode 16 /path/to/fakeapp.x --testid base2',
    #    'mpirun -host node-1,node-2 -np 32 -npernode 16 --xarg 1 /path/to/fakeapp.x --testid base3',
    #    'mpirun -host node-1,node-2 -np 128 -npernode 64 --xarg 1 /path/to/fakeapp.x --testid base4',
    #    'mpirun -host node-1 -np 16 -npernode 16 --xarg 1 /path/to/fakeapp.x --testid base5',
    #    'mpirun -host node-1,node-2 -np 16 -npernode 8 --xarg 1 /path/to/fakeapp.x --testid base6',
    #    'mpirun -host node-1 -np 16 --xarg 1 -npernode 16 /path/to/fakeapp.x --testid ompi1',
    #    'mpirun -host node-1,node-2 -np 8 --xarg 1 -npernode 4 /path/to/fakeapp.x --testid ompi2',
    #    ]

    exp_aprun = [
        "aprun -L node-1 -n 2 -N 2 --xarg 1 /path/to/fakeapp.x --testid base1",
        "aprun -L node-1,node-2 -n 32 -N 16 /path/to/fakeapp.x --testid base2",
        "aprun -L node-1,node-2 -n 32 -N 16 --xarg 1 /path/to/fakeapp.x --testid base3",
        "aprun -L node-1,node-2 -n 128 -N 64 --xarg 1 /path/to/fakeapp.x --testid base4",
        "aprun -L node-1 -n 16 -N 16 --xarg 1 /path/to/fakeapp.x --testid base5",
        "aprun -L node-1,node-2 -n 16 -N 8 --xarg 1 /path/to/fakeapp.x --testid base6",
        "aprun -L node-1 -n 16 --xarg 1 -N 16 /path/to/fakeapp.x --testid ap1",
        "aprun -L node-1,node-2 -n 8 --xarg 1 -N 4 /path/to/fakeapp.x --testid ap2",
    ]

    exp_srun = [
        "srun -w node-1 --ntasks 2 --nodes 1 --ntasks-per-node 2 --xarg 1 --exact /path/to/fakeapp.x --testid base1",
        "srun -w node-1,node-2 --ntasks 32 --nodes 2 --ntasks-per-node 16 --exact /path/to/fakeapp.x --testid base2",
        "srun -w node-1,node-2 --ntasks 32 --nodes 2 --ntasks-per-node 16 --xarg 1 --exact /path/to/fakeapp.x --testid base3",
        "srun -w node-1,node-2 --ntasks 128 --nodes 2 --ntasks-per-node 64 --xarg 1 --exact /path/to/fakeapp.x --testid base4",
        "srun -w node-1 --ntasks 16 --nodes 1 --ntasks-per-node 16 --xarg 1 --exact /path/to/fakeapp.x --testid base5",
        "srun -w node-1,node-2 --ntasks 16 --nodes 2 --ntasks-per-node 8 --xarg 1 --exact /path/to/fakeapp.x --testid base6",
        "srun -w node-1 --ntasks 16 --nodes 1 --xarg 1 --ntasks-per-node 16 --exact /path/to/fakeapp.x --testid sr1",
        "srun -w node-1,node-2 --nodes 2 -n 8 --xarg 1 --ntasks-per-node 4 --exact /path/to/fakeapp.x --testid sr2",
        "srun -w node-1,node-2 --nodes 2 -n 8 --xarg 1 --ntasks-per-node 4 --exact /path/to/fakeapp.x --testid sr3",
        "srun -w node-1,node-2 --nodes 2 -n 8 --xarg 1 --ntasks-per-node 4 --gpus-per-task 1 --exact /path/to/fakeapp.x --testid sr4",
        "srun -w node-1,node-2 --nodes 2 -n 8 --xarg 1 --ntasks-per-node 4 --gpus-per-task 1 --exact /path/to/fakeapp.x --testid sr5",
    ]

    exp_jsrun = [
        "jsrun -n 2 -r 2 --xarg 1 /path/to/fakeapp.x --testid base1",
        "jsrun -n 32 /path/to/fakeapp.x --testid base2",
        "jsrun -n 32 --xarg 1 /path/to/fakeapp.x --testid base3",
        "jsrun -n 128 --xarg 1 /path/to/fakeapp.x --testid base4",
        "jsrun -n 16 -r 16 --xarg 1 /path/to/fakeapp.x --testid base5",
        "jsrun -n 16 -r 8 --xarg 1 /path/to/fakeapp.x --testid base6",
        "jsrun -n 16 --xarg 1 -r 16 /path/to/fakeapp.x --testid jsr1",
        "jsrun -n 8 --xarg 1 -r 4 /path/to/fakeapp.x --testid jsr2",
        'jsrun -n 3 -r 3 -a 1 -c 1 -g 1 --bind=packed:1 --smpiargs="-gpu" /path/to/fakeapp.x --testid jsr3',
        'jsrun -r 3 -n 3 -a 1 -c 1 -g 1 --bind=packed:1 --smpiargs="-gpu" /path/to/fakeapp.x --testid jsr4',
    ]

    exp_custom = [
        "myrunner --xarg 1 /path/to/fakeapp.x --testid base1",
        "myrunner /path/to/fakeapp.x --testid base2",
        "myrunner --xarg 1 /path/to/fakeapp.x --testid base3",
        "myrunner --xarg 1 /path/to/fakeapp.x --testid base4",
        "myrunner --xarg 1 /path/to/fakeapp.x --testid base5",
        "myrunner --xarg 1 /path/to/fakeapp.x --testid base6",
        "myrunner --xarg 1 --ppn 16 /path/to/fakeapp.x --testid cust1",
    ]

    # Loop here for mocking different systems.
    def run_tests(mpi_runner, runner_name, test_list_exargs, exp_list):
        mpi_customizer = {
            "mpi_runner": mpi_runner,  # Select runner: mpich, openmpi, aprun, srun, jsrun
            "runner_name": runner_name,  # Runner name: Replaces run command if not None
        }

        exctr = MPIExecutor(custom_info=mpi_customizer)
        exctr.register_app(full_path=sim_app, calc_type="sim")

        test_list = test_list_base + test_list_exargs
        sim_specs["user"] = {
            "expect": exp_list,
            "nodes_per_worker": nodes_per_worker,
            "offset_for_scheduler": True,
            "persis_gens": 0,
            "tests": test_list,
        }

        # Perform the run
        H, pinfo, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    # for run_set in ['mpich', 'openmpi', 'aprun', 'srun', 'jsrun', 'rename_mpich', 'custom']:
    for run_set in ["mpich", "aprun", "srun", "jsrun", "rename_mpich", "custom"]:
        # Could use classes, pref in separate data_set module
        runner_name = None  # Use default

        if run_set == "mpich":
            runner = "mpich"
            test_list_exargs = eargs_mpich
            exp_list = exp_mpich

        #    if run_set == 'openmpi':
        #        runner = 'openmpi'
        #        test_list_exargs = eargs_openmpi
        #        exp_list = exp_openmpi

        if run_set == "aprun":
            runner = "aprun"
            test_list_exargs = eargs_aprun
            exp_list = exp_aprun

        if run_set == "srun":
            runner = "srun"
            test_list_exargs = eargs_srun
            exp_list = exp_srun

        if run_set == "jsrun":
            runner = "jsrun"
            test_list_exargs = eargs_jsrun
            exp_list = exp_jsrun

        if run_set == "rename_mpich":
            runner = "mpich"
            runner_name = "inst -dummy mpich"
            test_list_exargs = eargs_mpich
            exp_list = exp_rename_mpich

        # Auto-resources will not parse - everything goes in extra_args
        if run_set == "custom":
            runner = "custom"
            runner_name = "myrunner"
            test_list_exargs = eargs_custom
            exp_list = exp_custom

        run_tests(runner, runner_name, test_list_exargs, exp_list)

    # All asserts are in sim func
