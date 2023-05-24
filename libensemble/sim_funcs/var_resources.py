"""
Simulation functions that use the MPIExecutor with dynamic resource assignment.
``six_hump_camel`` and ``helloworld`` python scripts are used as example
applications, but these could be any MPI application.

Each simulation function use the resources assigned to this worker to set CPU
count and, in some functions, specify GPU usage.

GPUs are not used for the six_hump_camel function, but these tests check the
assignment is correct. For an example that runs an actual GPU application, see
the forces_gpu tutorial under libensemble/tests/scaling_tests/forces/forces_gpu.

See CUDA_variable_resources for an example where the sim function
interrogates available resources and sets explicitly.

"""

__all__ = [
    "gpu_variable_resources",
    "gpu_variable_resources_from_gen",
    "multi_points_with_variable_resources",
    "CUDA_variable_resources",
]

import os
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import TASK_FAILED, UNSET_TAG, WORKER_DONE
from libensemble.resources.resources import Resources
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func
from libensemble.tools.test_support import check_gpu_setting


def gpu_variable_resources(H, persis_info, sim_specs, libE_info):
    """Launches an app and automatically assigns GPU resources.

    The six_hump_camel app does not run on the GPU, but this test demonstrates
    how to automatically assign the GPUs given to this worker via the MPIExecutor.

    The method used to assign GPUs will be determined by the MPI runner or user
    provided configuration (e.g., by setting the ``platform`` or ``platform_specs``
    options or the LIBE_PLATFORM environment variable).

    """
    x = H["x"][0]
    H_o = np.zeros(1, dtype=sim_specs["out"])
    dry_run = sim_specs["user"].get("dry_run", False)  # logs run lines instead of running
    inpt = " ".join(map(str, x))  # Application input

    exctr = Executor.executor  # Get Executor

    # Launch application via system MPI runner, using assigned resources.
    task = exctr.submit(
        app_name="six_hump_camel",
        app_args=inpt,
        auto_assign_gpus=True,
        match_procs_to_gpus=True,
        stdout="out.txt",
        stderr="err.txt",
        dry_run=dry_run,
    )

    if not dry_run:
        task.wait()  # Wait for run to complete

        # Access app output
        with open("out.txt") as f:
            H_o["f"] = float(f.readline().strip())  # Read just first line

    # Asserts GPU set correctly (for known MPI runners)
    check_gpu_setting(task, print_setting=True)

    calc_status = WORKER_DONE if task.state == "FINISHED" else "FAILED"
    return H_o, persis_info, calc_status


def gpu_variable_resources_from_gen(H, persis_info, sim_specs, libE_info):
    """Launches an app and assigns CPU and GPU resources as defined by the gen.

    Otherwise similar to gpu_variable_resources.
    """
    x = H["x"][0]
    H_o = np.zeros(1, dtype=sim_specs["out"])
    dry_run = sim_specs["user"].get("dry_run", False)  # logs run lines instead of running
    inpt = " ".join(map(str, x))  # Application input

    exctr = Executor.executor  # Get Executor

    # Launch application via system MPI runner, using assigned resources.
    task = exctr.submit(
        app_name="six_hump_camel",
        app_args=inpt,
        stdout="out.txt",
        stderr="err.txt",
        dry_run=dry_run,
    )

    if not dry_run:
        task.wait()  # Wait for run to complete

        # Access app output
        with open("out.txt") as f:
            H_o["f"] = float(f.readline().strip())  # Read just first line

    # Asserts GPU set correctly (for known MPI runners)
    check_gpu_setting(task, print_setting=True)

    calc_status = WORKER_DONE if task.state == "FINISHED" else "FAILED"
    return H_o, persis_info, calc_status


def multi_points_with_variable_resources(H, _, sim_specs):
    """
    Evaluates either helloworld or six hump camel for a collection of points
    given in ``H["x"]`` via the MPI executor, supporting variable sized
    simulations/resources, as determined by the generator. The term `rset`
    refers to a resource set (the minimal set of resources that can be assigned
    to each worker). It can be anything from a partition of a node to multiple
    nodes.

    Note that this is also an example that is capable of handling multiple
    points (sim ids) in one each call.

    .. seealso::
        `test_uniform_sampling_with_variable_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_with_variable_resources.py>`_ # noqa
    """

    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])
    app = sim_specs["user"].get("app", "helloworld")
    dry_run = sim_specs["user"].get("dry_run", False)  # dry_run only prints run lines in ensemble.log
    set_cores_by_rsets = True  # If True use rset count to set num procs, else use all available to this worker.
    core_multiplier = 1  # Only used with set_cores_by_rsets as a multiplier.

    exctr = Executor.executor  # Get Executor
    task_states = []
    for i, x in enumerate(H["x"]):
        nprocs = None  # Will be as if argument is not present
        if set_cores_by_rsets:
            resources = Resources.resources.worker_resources
            nprocs = resources.num_rsets * core_multiplier

        inpt = None  # Will be as if argument is not present
        if app == "six_hump_camel":
            inpt = " ".join(map(str, H["x"][i]))

        print(f"{nprocs=}")

        task = exctr.submit(
            app_name=app,
            app_args=inpt,
            num_procs=nprocs,
            stdout="out.txt",
            stderr="err.txt",
            dry_run=dry_run,
        )

        if not dry_run:
            task.wait()  # Wait for run to complete

            # while(not task.finished):
            #     time.sleep(0.1)
            #     task.poll()

        task_states.append(task.state)

        if app == "six_hump_camel":
            # H_o["f"][i] = float(task.read_stdout())  # Reads whole file
            with open("out.txt") as f:
                H_o["f"][i] = float(f.readline().strip())  # Read just first line
        else:
            # To return something in test
            H_o["f"][i] = six_hump_camel_func(x)

    calc_status = UNSET_TAG  # Returns to worker
    if all(t == "FINISHED" for t in task_states):
        calc_status = WORKER_DONE
    elif any(t == "FAILED" for t in task_states):
        calc_status = TASK_FAILED

    return H_o, calc_status


def CUDA_variable_resources(H, _, sim_specs, libE_info):
    """Launches an app setting GPU resources

    The standard test apps do not run on GPU, but demonstrates accessing resource
    information to set ``CUDA_VISIBLE_DEVICES``, and typical run configuration.

    For an equivalent function that auto-assigns GPUs using platform detection, see
    GPU_variable_resources.
    """
    x = H["x"][0]
    H_o = np.zeros(1, dtype=sim_specs["out"])
    dry_run = sim_specs["user"].get("dry_run", False)  # dry_run only prints run lines in ensemble.log

    # Interrogate resources available to this worker
    resources = Resources.resources.worker_resources
    slots = resources.slots

    assert resources.matching_slots, f"Error: Cannot set CUDA_VISIBLE_DEVICES when unmatching slots on nodes {slots}"

    num_nodes = resources.local_node_count

    # Set to slots
    resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")
    cores_per_node = resources.slot_count

    # Set to detected GPUs
    # gpus_per_slot = resources.gpus_per_rset
    # resources.set_env_to_slots("CUDA_VISIBLE_DEVICES", multiplier=gpus_per_slot)
    # cores_per_node = resources.slot_count * gpus_per_slot  # One CPU per GPU

    print(
        f"Worker {libE_info['workerID']}: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
        f"\tnodes {num_nodes} ppn {cores_per_node}  slots {slots}"
    )

    # Create application input file
    inpt = " ".join(map(str, x))
    exctr = Executor.executor  # Get Executor

    # Launch application via system MPI runner, using assigned resources.
    task = exctr.submit(
        app_name="six_hump_camel",
        app_args=inpt,
        num_nodes=num_nodes,
        procs_per_node=cores_per_node,
        stdout="out.txt",
        stderr="err.txt",
        dry_run=dry_run,
        # extra_args='--gpus-per-task=1'
    )

    if not dry_run:
        task.wait()  # Wait for run to complete

        # Access app output
        with open("out.txt") as f:
            H_o["f"] = float(f.readline().strip())  # Read just first line

    calc_status = WORKER_DONE if task.state == "FINISHED" else "FAILED"
    return H_o, calc_status
