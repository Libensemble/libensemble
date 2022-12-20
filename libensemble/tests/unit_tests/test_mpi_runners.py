import sys
import pytest
import numpy as np
from libensemble import logger
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.tests.regression_tests.common import create_node_file


@pytest.mark.extra
def test_basic():
    from libensemble.sim_funcs.run_line_check import runline_check as sim_f
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers, "dedicated_mode": True, "enforce_worker_core_bounds": True}

    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    log_file = "ensemble_mpi_runners_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    nodes_per_worker = 2

    # For varying size test - relate node count to nworkers
    node_file = "nodelist_mpi_runners_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = nworkers * nodes_per_worker

    create_node_file(num_nodes=nnodes, name=node_file)

    # Mock up system
    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,  # Name of file containing a node-list
    }

    libE_specs["resource_info"] = custom_resources

    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"sim_max": nworkers}

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

    # eargs_openmpi = [
    #     {"testid": "ompi1", "nprocs": 16, "e_args": "--xarg 1 -npernode 16"},  # nprocs + parse extra_args
    #     {"testid": "ompi2", "e_args": "-np 8 --xarg 1 -npernode 4"},  # parse extra_args
    # ]

    eargs_aprun = [
        {"testid": "ap1", "nprocs": 16, "e_args": "--xarg 1 -N 16"},  # nprocs + parse extra_args
        {"testid": "ap2", "e_args": "-n 8 --xarg 1 -N 4"},  # parse extra_args
    ]

    # Note in a8: -n 8 is abbreviated form of --ntasks, this should be unaltered while --nodes is derived and inserted.
    eargs_srun = [
        {"testid": "sr1", "nprocs": 16, "e_args": "--xarg 1 --ntasks-per-node 16"},  # nprocs + parse extra_args
        {"testid": "sr2", "e_args": "-n 8 --xarg 1 --ntasks-per-node 4"},  # parse extra_args
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
        "srun -w node-1 --ntasks 2 --nodes 1 --ntasks-per-node 2 --xarg 1 /path/to/fakeapp.x --testid base1",
        "srun -w node-1,node-2 --ntasks 32 --nodes 2 --ntasks-per-node 16 /path/to/fakeapp.x --testid base2",
        "srun -w node-1,node-2 --ntasks 32 --nodes 2 --ntasks-per-node 16 --xarg 1 /path/to/fakeapp.x --testid base3",
        "srun -w node-1,node-2 --ntasks 128 --nodes 2 --ntasks-per-node 64 --xarg 1 /path/to/fakeapp.x --testid base4",
        "srun -w node-1 --ntasks 16 --nodes 1 --ntasks-per-node 16 --xarg 1 /path/to/fakeapp.x --testid base5",
        "srun -w node-1,node-2 --ntasks 16 --nodes 2 --ntasks-per-node 8 --xarg 1 /path/to/fakeapp.x --testid base6",
        "srun -w node-1 --ntasks 16 --nodes 1 --xarg 1 --ntasks-per-node 16 /path/to/fakeapp.x --testid sr1",
        "srun -w node-1,node-2 --nodes 2 -n 8 --xarg 1 --ntasks-per-node 4 /path/to/fakeapp.x --testid sr2",
    ]

    exp_jsrun = [
        "jsrun -n 2 -r 2 --xarg 1 /path/to/fakeapp.x --testid base1",
        "jsrun -n 32 /path/to/fakeapp.x --testid base2",
        "jsrun -n 32 --xarg 1 /path/to/fakeapp.x --testid base3",
        "jsrun -n 128 --xarg 1 /path/to/fakeapp.x --testid base4",
        "jsrun -n 16 --xarg 1 /path/to/fakeapp.x --testid base5",
        "jsrun -n 16 -r 8 --xarg 1 /path/to/fakeapp.x --testid base6",
        "jsrun -n 16 --xarg 1 -r 16 /path/to/fakeapp.x --testid jsr1",
        "jsrun -n 8 --xarg 1 -r 4 /path/to/fakeapp.x --testid jsr2",
        'jsrun -n 3 -a 1 -c 1 -g 1 --bind=packed:1 --smpiargs="-gpu" /path/to/fakeapp.x --testid jsr3',
        'jsrun -n 3 -a 1 -c 1 -g 1 --bind=packed:1 --smpiargs="-gpu" /path/to/fakeapp.x --testid jsr4',
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

        H, pinfo, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    # for run_set in ['mpich', 'openmpi', 'aprun', 'srun', 'jsrun', 'rename_mpich', 'custom']:
    for run_set in ["mpich", "aprun", "srun", "jsrun", "rename_mpich", "custom"]:

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


@pytest.mark.extra
def test_subnode_uneven():

    from libensemble.sim_funcs.run_line_check import runline_check_by_worker as sim_f
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

    nworkers = 5
    libE_specs = {"comms": "local", "nworkers": nworkers, "dedicated_mode": True, "enforce_worker_core_bounds": True}

    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    log_file = "ensemble_mpi_runners_subnode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    # For varying size test - relate node count to nworkers
    nsim_workers = nworkers

    if nsim_workers % 2 == 0:
        sys.exit(f"This test must be run with an odd of workers >= 3 and <= 31. There are {nsim_workers} workers.")

    comms = libE_specs["comms"]
    node_file = "nodelist_mpi_runners_subnode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = 2

    create_node_file(num_nodes=nnodes, name=node_file)

    # Mock up system
    mpi_customizer = {
        "mpi_runner": "srun",  # Select runner: mpich, openmpi, aprun, srun, jsrun
        "runner_name": "srun",
    }  # Runner name: Replaces run command if not None

    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,
    }  # Name of file containing a node-list

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
    # [w1]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
    # [w2]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
    # [w3]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
    # [w4]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base1
    # [w5]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base1

    srun_p1 = "srun -w "
    srun_p2 = " --ntasks "
    srun_p3 = " --nodes 1 --ntasks-per-node "
    srun_p4 = " /path/to/fakeapp.x --testid base1"

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

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)


@pytest.mark.extra
def test_subnode():
    from libensemble.sim_funcs.run_line_check import runline_check as sim_f
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers, "dedicated_mode": True, "enforce_worker_core_bounds": True}
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

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

    create_node_file(num_nodes=nnodes, name=node_file)

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
        "srun -w node-1 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base1",
        "srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base2",
        "srun -w node-1 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base3",
        "srun -w node-1 --ntasks 6 --nodes 1 --ntasks-per-node 6 /path/to/fakeapp.x --testid base4",
    ]

    test_list = test_list_base
    exp_list = exp_srun
    sim_specs["user"] = {
        "tests": test_list,
        "expect": exp_list,
        "nodes_per_worker": nodes_per_worker,
    }

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)


@pytest.mark.extra
def test_supernode_uneven():
    from libensemble.sim_funcs.run_line_check import runline_check_by_worker as sim_f
    from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers, "dedicated_mode": True, "enforce_worker_core_bounds": True}
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    # To allow visual checking - log file not used in test
    log_file = "ensemble_mpi_runners_supernode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    nodes_per_worker = 2.5

    # For varying size test - relate node count to nworkers
    nsim_workers = nworkers
    comms = libE_specs["comms"]
    node_file = "nodelist_mpi_runners_supernode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = int(nsim_workers * nodes_per_worker)

    # Mock up system
    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,
    }  # Name of file containing a node-list
    libE_specs["resource_info"] = custom_resources

    create_node_file(num_nodes=nnodes, name=node_file)

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

    # Each worker has either 3 or 2 nodes. Basic test list for portable options
    test_list_base = [
        {"testid": "base1"},  # Give no config and no extra_args
    ]

    # Example: On 2 workers, runlines should be ...
    # (one workers has 3 nodes, the other 2 - does not split 2.5 nodes each).
    # [w1]: srun -w node-1,node-2,node-3 --ntasks 48 --nodes 3 --ntasks-per-node 16 /path/to/fakeapp.x --testid base1
    # [w2]: srun -w node-4,node-5 --ntasks 32 --nodes 2 --ntasks-per-node 16 /path/to/fakeapp.x --testid base1

    srun_p1 = "srun -w "
    srun_p2 = " --ntasks "
    srun_p3 = " --nodes "
    srun_p4 = " --ntasks-per-node 16 /path/to/fakeapp.x --testid base1"

    exp_tasks = []
    exp_srun = []

    # Hard coding an example for 2 nodes to avoid replicating general logic in libEnsemble.
    low_npw = nnodes // nsim_workers
    high_npw = nnodes // nsim_workers + 1

    nodelist = []
    for i in range(1, nnodes + 1):
        nodelist.append("node-" + str(i))

    inode = 0
    for i in range(nsim_workers):
        if i < (nsim_workers // 2):
            npw = high_npw
        else:
            npw = low_npw
        nodename = ",".join(nodelist[inode : inode + npw])
        inode += npw
        ntasks = 16 * npw
        loc_nodes = npw
        exp_tasks.append(ntasks)
        exp_srun.append(srun_p1 + str(nodename) + srun_p2 + str(ntasks) + srun_p3 + str(loc_nodes) + srun_p4)

    test_list = test_list_base
    exp_list = exp_srun
    sim_specs["user"] = {
        "tests": test_list,
        "expect": exp_list,
    }

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)


@pytest.mark.extra
def test_zrw_subnode_uneven():
    from libensemble.sim_funcs.run_line_check import runline_check_by_worker as sim_f
    from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
    from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f

    nworkers = 4
    libE_specs = {"comms": "local", "nworkers": nworkers, "dedicated_mode": True, "enforce_worker_core_bounds": True}
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    # To allow visual checking - log file not used in test
    log_file = "ensemble_mpi_runners_zrw_subnode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    # For varying size test - relate node count to nworkers
    n_gens = 1
    nsim_workers = nworkers - n_gens

    if nsim_workers % 2 == 0:
        sys.exit(
            "This test must be run with an odd number of sim workers >= 3 and <= 31. There are {} sim workers.".format(
                nsim_workers
            )
        )

    comms = libE_specs["comms"]
    node_file = "nodelist_mpi_runners_zrw_subnode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = 2

    # Mock up system
    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,
    }  # Name of file containing a node-list
    libE_specs["resource_info"] = custom_resources

    create_node_file(num_nodes=nnodes, name=node_file)

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
            "initial_batch_size": 20,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}
    exit_criteria = {"sim_max": (nsim_workers) * rounds}

    test_list_base = [
        {"testid": "base1"},  # Give no config and no extra_args
    ]

    # Example: On 5 workers, runlines should be ...
    # [w1]: Gen only
    # [w2]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
    # [w3]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
    # [w4]: srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 /path/to/fakeapp.x --testid base1
    # [w5]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base1
    # [w6]: srun -w node-2 --ntasks 8 --nodes 1 --ntasks-per-node 8 /path/to/fakeapp.x --testid base1

    srun_p1 = "srun -w "
    srun_p2 = " --ntasks "
    srun_p3 = " --nodes 1 --ntasks-per-node "
    srun_p4 = " /path/to/fakeapp.x --testid base1"

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
        "persis_gens": n_gens,
    }

    iterations = 2
    for prob_id in range(iterations):

        if prob_id == 0:
            # Uses dynamic scheduler - will find node 2 slots first (as fewer)
            libE_specs["num_resource_sets"] = nworkers - 1  # Any worker can be the gen
            sim_specs["user"]["offset_for_scheduler"] = True  # Changes expected values
            persis_info = add_unique_random_streams({}, nworkers + 1)

        else:
            # Uses static scheduler - will find node 1 slots first
            del libE_specs["num_resource_sets"]
            libE_specs["zero_resource_workers"] = [1]  # Gen must be worker 1
            sim_specs["user"]["offset_for_scheduler"] = False
            persis_info = add_unique_random_streams({}, nworkers + 1)

        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)


@pytest.mark.extra
def test_zrw_supernode_uneven():
    from libensemble.sim_funcs.run_line_check import runline_check_by_worker as sim_f
    from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
    from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f

    nworkers = 4
    libE_specs = {
        "comms": "local",
        "nworkers": nworkers,
        "zero_resource_workers": [1],
        "dedicated_mode": True,
        "enforce_worker_core_bounds": True,
    }
    rounds = 1
    sim_app = "/path/to/fakeapp.x"
    comms = libE_specs["comms"]

    # To allow visual checking - log file not used in test
    log_file = "ensemble_mpi_runners_zrw_supernode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers) + ".log"
    logger.set_filename(log_file)

    nodes_per_worker = 2.5

    # For varying size test - relate node count to nworkers
    in_place = libE_specs["zero_resource_workers"]
    n_gens = len(in_place)
    nsim_workers = nworkers - n_gens
    comms = libE_specs["comms"]
    node_file = "nodelist_mpi_runners_zrw_supernode_uneven_comms_" + str(comms) + "_wrks_" + str(nworkers)
    nnodes = int(nsim_workers * nodes_per_worker)

    # Mock up system
    custom_resources = {
        "cores_on_node": (16, 64),  # Tuple (physical cores, logical cores)
        "node_file": node_file,  # Name of file containing a node-list
    }
    libE_specs["resource_info"] = custom_resources

    create_node_file(num_nodes=nnodes, name=node_file)

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
            "initial_batch_size": 20,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}
    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"sim_max": (nsim_workers) * rounds}

    # Each worker has either 3 or 2 nodes. Basic test list for portable options
    test_list_base = [
        {"testid": "base1"},  # Give no config and no extra_args
    ]

    # Example: On 3 workers, runlines should be ...
    # (one workers has 3 nodes, the other 2 - does not split 2.5 nodes each).
    # [w1]: Gen only
    # [w2]: srun -w node-1,node-2,node-3 --ntasks 48 --nodes 3 --ntasks-per-node 16 /path/to/fakeapp.x --testid base1
    # [w3]: srun -w node-4,node-5 --ntasks 32 --nodes 2 --ntasks-per-node 16 /path/to/fakeapp.x --testid base1

    srun_p1 = "srun -w "
    srun_p2 = " --ntasks "
    srun_p3 = " --nodes "
    srun_p4 = " --ntasks-per-node 16 /path/to/fakeapp.x --testid base1"

    exp_tasks = []
    exp_srun = []

    low_npw = nnodes // nsim_workers
    high_npw = nnodes // nsim_workers + 1

    nodelist = []
    for i in range(1, nnodes + 1):
        nodelist.append("node-" + str(i))

    inode = 0
    for i in range(nsim_workers):
        if i < (nsim_workers // 2):
            npw = high_npw
        else:
            npw = low_npw
        nodename = ",".join(nodelist[inode : inode + npw])
        inode += npw
        ntasks = 16 * npw
        loc_nodes = npw
        exp_tasks.append(ntasks)
        exp_srun.append(srun_p1 + str(nodename) + srun_p2 + str(ntasks) + srun_p3 + str(loc_nodes) + srun_p4)

    test_list = test_list_base
    exp_list = exp_srun
    sim_specs["user"] = {
        "tests": test_list,
        "expect": exp_list,
        "persis_gens": n_gens,
    }

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)


if __name__ == "__main__":
    test_basic()
    test_subnode_uneven()
    test_subnode()
    test_supernode_uneven()
    test_zrw_subnode_uneven()
    test_zrw_supernode_uneven()
