# !/usr/bin/env python
import os
import sys

from libensemble.executors.executor import Executor
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.resources.resources import Resources
from libensemble.resources.platforms import get_platform


NCORES = 1
build_sims = ["my_simtask.c", "my_serialtask.c", "c_startup.c"]

sim_app = "simdir/my_simtask.x"


def setup_module(module):
    try:
        print(f"setup_module module:{module.__name__}")
    except AttributeError:
        print(f"setup_module (direct run) module:{module}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None
    build_simfuncs()


def setup_function(function):
    print(f"setup_function function:{function.__name__}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def teardown_module(module):
    try:
        print(f"teardown_module module:{module.__name__}")
    except AttributeError:
        print(f"teardown_module (direct run) module:{module}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def build_simfuncs():
    import subprocess

    for sim in build_sims:
        app_name = ".".join([sim.split(".")[0], "x"])
        if not os.path.isfile(app_name):
            buildstring = "mpicc -o " + os.path.join("simdir", app_name) + " " + os.path.join("simdir", sim)
            subprocess.check_call(buildstring.split())


def setup_exe_from_platform(rset_team, nworkers, workerID, mpi_runner, platform, nodes):
    """Sets up resources and executor from platform.

    Also sets up workers resources and mocks being on a worker.
    """

    plat1 = {
        "mpi_runner": mpi_runner,
        "cores_per_node": 64,
        "logical_cores_per_node": 128,
        "gpus_per_node": 8,
        "gpu_setting_type": "runner_default",
        "scheduler_match_slots": True,
    }

    plat2 = {
        "mpi_runner": mpi_runner,
        "cores_per_node": 64,
        "logical_cores_per_node": 128,
        "gpus_per_node": 8,
        "gpu_setting_type": "option_gpus_per_node",
        "gpu_setting_name": "--gpus_on_this_node=",
        "scheduler_match_slots": False,
    }

    plat3 = {
        "mpi_runner": mpi_runner,
        "cores_per_node": 64,
        "logical_cores_per_node": 128,
        "gpus_per_node": 8,
        "gpu_setting_type": "env",
        "gpu_setting_name": "TESTING_VISIBLE_DEVICES",
        "scheduler_match_slots": False,
    }

    platforms = {"plat1": plat1, "plat2": plat2, "plat3": plat3}

    plat_specs = platforms[platform]

    if nodes == 1:
        os.environ["LIBE_EXECUTOR_TEST_NODE_LIST"] = "node-1"
    elif nodes == 2:
        os.environ["LIBE_EXECUTOR_TEST_NODE_LIST"] = "node-[1-2]"

    resource_info = {"nodelist_env_slurm": "LIBE_EXECUTOR_TEST_NODE_LIST"}

    libE_specs = {"platform_specs": plat_specs, "num_resource_sets": 8, "resource_info": resource_info}

    platform_info = get_platform(libE_specs)
    Resources.init_resources(libE_specs, platform_info)
    resources = Resources.resources
    resources.set_worker_resources(nworkers, workerID)
    resources.worker_resources.set_rset_team(rset_team)

    exctr = MPIExecutor()
    exctr.add_platform_info(platform_info)
    exctr.register_app(full_path=sim_app, calc_type="sim")
    exctr = Executor.executor
    exctr.set_resources(resources)
    return exctr


def set_extr_check(exctr):
    def run_check(exp_env, exp_cmd, **kwargs):
        args_for_sim = "sleep 0"
        exp_runline = exp_cmd + " simdir/my_simtask.x sleep 0"
        task = exctr.submit(calc_type="sim", app_args=args_for_sim, dry_run=True, **kwargs)
        assert task.env == exp_env, f"task.env does not match expected: {task.env}"
        assert task.runline == exp_runline, f"exp_runline does not match expected: {task.runline}"

    return run_check


def test_dry_run_ngpus():
    """Test setting of GPUs and runlines using dryrun"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    platform = "plat1"
    mpi_runner = "mpich"
    nodes = 1
    nworkers = 1
    workerID = 1
    rset_team = [0, 2, 3, 4, 6]  # this worker has 5 of 8 slots
    exctr = setup_exe_from_platform(rset_team, nworkers, workerID, mpi_runner, platform, nodes)
    run_check = set_extr_check(exctr)

    # auto_assign_gpus
    exp_env = {"CUDA_VISIBLE_DEVICES": "0,2,3,4,6"}
    exp_cmd = "mpirun -hosts node-1 -np 1 --ppn 1"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True)
    run_check(exp_env, exp_cmd, procs_per_node=1, auto_assign_gpus=True)

    # restrict with num_gpus - too many, restrict to those available
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=10)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=10)

    # restrict with num_gpus
    exp_env = {"CUDA_VISIBLE_DEVICES": "0,2"}
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=2)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=2)

    # specify nodes
    run_check(exp_env, exp_cmd, num_procs=1, num_nodes=1, num_gpus=2)

    # match_procs_to_gpus
    exp_env = {"CUDA_VISIBLE_DEVICES": "0,2,3,4,6"}
    exp_cmd = "mpirun -hosts node-1 -np 5 --ppn 5"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, auto_assign_gpus=True)

    exp_env = {"CUDA_VISIBLE_DEVICES": "0,2,3"}
    exp_cmd = "mpirun -hosts node-1 -np 3 --ppn 3"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, num_gpus=3)


def test_dry_run_ngpus_srun():
    """Test setting of GPUs and runlines using dryrun"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    platform = "plat1"
    mpi_runner = "srun"
    nodes = 1
    nworkers = 1
    workerID = 1
    rset_team = [0, 2, 3, 4, 6]  # this worker has 5 of 8 slots
    exctr = setup_exe_from_platform(rset_team, nworkers, workerID, mpi_runner, platform, nodes)
    run_check = set_extr_check(exctr)

    # auto_assign_gpus
    exp_env = {}

    exp_cmd = "srun -w node-1 --ntasks 1 --nodes 1 --ntasks-per-node 1 --gpus-per-node 5 --exact"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True)
    run_check(exp_env, exp_cmd, procs_per_node=1, auto_assign_gpus=True)

    # restrict with num_gpus - too many, restrict to those available
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=10)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=10)

    # restrict with num_gpus
    exp_cmd = "srun -w node-1 --ntasks 1 --nodes 1 --ntasks-per-node 1 --gpus-per-node 2 --exact"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=2)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=2)

    # specify nodes
    run_check(exp_env, exp_cmd, num_procs=1, num_nodes=1, num_gpus=2)

    # match_procs_to_gpus
    exp_cmd = "srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 --gpus-per-node 5 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, auto_assign_gpus=True)

    exp_cmd = "srun -w node-1 --ntasks 3 --nodes 1 --ntasks-per-node 3 --gpus-per-node 3 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, num_gpus=3)


def test_dry_run_ngpus_srun_plat2():
    """Test setting of GPUs and runlines using dryrun"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    platform = "plat2"
    mpi_runner = "srun"
    nodes = 1
    nworkers = 1
    workerID = 1
    rset_team = [0, 2, 3, 4, 6]  # this worker has 5 of 8 slots
    exctr = setup_exe_from_platform(rset_team, nworkers, workerID, mpi_runner, platform, nodes)
    run_check = set_extr_check(exctr)

    # auto_assign_gpus
    exp_env = {}

    exp_cmd = "srun -w node-1 --ntasks 1 --nodes 1 --ntasks-per-node 1 --gpus_on_this_node=5 --exact"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True)
    run_check(exp_env, exp_cmd, procs_per_node=1, auto_assign_gpus=True)

    # restrict with num_gpus - too many, restrict to those available
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=10)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=10)

    # restrict with num_gpus
    exp_cmd = "srun -w node-1 --ntasks 1 --nodes 1 --ntasks-per-node 1 --gpus_on_this_node=2 --exact"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=2)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=2)

    # specify nodes
    run_check(exp_env, exp_cmd, num_procs=1, num_nodes=1, num_gpus=2)

    # match_procs_to_gpus
    exp_cmd = "srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 --gpus_on_this_node=5 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, auto_assign_gpus=True)

    exp_cmd = "srun -w node-1 --ntasks 3 --nodes 1 --ntasks-per-node 3 --gpus_on_this_node=3 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, num_gpus=3)


def test_dry_run_ngpus_srun_plat3():
    """Test setting of GPUs and runlines using dryrun"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    platform = "plat3"
    mpi_runner = "srun"
    nodes = 1
    nworkers = 1
    workerID = 1
    rset_team = [0, 2, 3, 4, 6]  # this worker has 5 of 8 slots
    exctr = setup_exe_from_platform(rset_team, nworkers, workerID, mpi_runner, platform, nodes)
    run_check = set_extr_check(exctr)

    # auto_assign_gpus
    exp_env = {"TESTING_VISIBLE_DEVICES": "0,2,3,4,6"}
    exp_cmd = "srun -w node-1 --ntasks 1 --nodes 1 --ntasks-per-node 1 --exact"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True)
    run_check(exp_env, exp_cmd, procs_per_node=1, auto_assign_gpus=True)

    # restrict with num_gpus - too many, restrict to those available
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=10)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=10)

    # restrict with num_gpus
    exp_env = {"TESTING_VISIBLE_DEVICES": "0,2"}
    exp_cmd = "srun -w node-1 --ntasks 1 --nodes 1 --ntasks-per-node 1 --exact"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=2)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=2)

    # specify nodes
    run_check(exp_env, exp_cmd, num_procs=1, num_nodes=1, num_gpus=2)

    # match_procs_to_gpus
    exp_env = {"TESTING_VISIBLE_DEVICES": "0,2,3,4,6"}
    exp_cmd = "srun -w node-1 --ntasks 5 --nodes 1 --ntasks-per-node 5 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, auto_assign_gpus=True)

    exp_env = {"TESTING_VISIBLE_DEVICES": "0,2,3"}
    exp_cmd = "srun -w node-1 --ntasks 3 --nodes 1 --ntasks-per-node 3 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, num_gpus=3)


def test_dry_run_ngpus_srun_plat3_2nodes():
    """Test setting of GPUs and runlines using dryrun"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    platform = "plat3"
    mpi_runner = "srun"
    nodes = 2
    nworkers = 1
    workerID = 1

    # Note - 8 resource sets, 8 gpus per node - 2 gpus per resource set.
    rset_team = [0, 1, 2, 4, 5, 6]  # this worker has 3 slots on each of 2 nodes.
    exctr = setup_exe_from_platform(rset_team, nworkers, workerID, mpi_runner, platform, nodes)
    run_check = set_extr_check(exctr)

    exp_env = {"TESTING_VISIBLE_DEVICES": "0,1,2,3,4,5"}
    exp_cmd = "srun -w node-1,node-2 --ntasks 2 --nodes 2 --ntasks-per-node 1 --exact"
    run_check(exp_env, exp_cmd, num_procs=2, num_nodes=2, auto_assign_gpus=True)
    run_check(exp_env, exp_cmd, procs_per_node=1, auto_assign_gpus=True)

    # auto_assign_gpus
    exp_env = {"TESTING_VISIBLE_DEVICES": "0,1,2,3,4,5"}
    exp_cmd = "srun -w node-1 --ntasks 1 --nodes 1 --ntasks-per-node 1 --exact"
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True)

    # restrict with num_gpus - too many, restrict to those available
    run_check(exp_env, exp_cmd, num_procs=1, auto_assign_gpus=True, num_gpus=10)
    run_check(exp_env, exp_cmd, num_procs=1, num_gpus=10)

    exp_env = {"TESTING_VISIBLE_DEVICES": "0,1,2,3,4,5"}
    exp_cmd = "srun -w node-1,node-2 --ntasks 2 --nodes 2 --ntasks-per-node 1 --exact"
    run_check(exp_env, exp_cmd, procs_per_node=1, auto_assign_gpus=True)

    # restrict with num_gpus
    exp_env = {"TESTING_VISIBLE_DEVICES": "0"}
    exp_cmd = "srun -w node-1,node-2 --ntasks 2 --nodes 2 --ntasks-per-node 1 --exact"
    run_check(exp_env, exp_cmd, num_procs=2, auto_assign_gpus=True, num_gpus=2)
    run_check(exp_env, exp_cmd, num_procs=2, num_gpus=2)

    # match_procs_to_gpus
    exp_env = {"TESTING_VISIBLE_DEVICES": "0,1,2,3,4,5"}
    exp_cmd = "srun -w node-1,node-2 --ntasks 12 --nodes 2 --ntasks-per-node 6 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, auto_assign_gpus=True)

    exp_env = {"TESTING_VISIBLE_DEVICES": "0,1"}
    exp_cmd = "srun -w node-1,node-2 --ntasks 4 --nodes 2 --ntasks-per-node 2 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, num_gpus=4)

    exp_env = {"TESTING_VISIBLE_DEVICES": "0"}
    exp_cmd = "srun -w node-1,node-2 --ntasks 2 --nodes 2 --ntasks-per-node 1 --exact"
    run_check(exp_env, exp_cmd, match_procs_to_gpus=True, num_gpus=3)


if __name__ == "__main__":
    setup_module(__file__)
    test_dry_run_ngpus()
    test_dry_run_ngpus_srun()
    test_dry_run_ngpus_srun_plat2()
    test_dry_run_ngpus_srun_plat3()
    test_dry_run_ngpus_srun_plat3_2nodes()
    teardown_module(__file__)
