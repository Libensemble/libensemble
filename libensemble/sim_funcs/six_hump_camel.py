"""
This module contains various versions that evaluate the six-hump camel function.

Six-hump camel function is documented here:
  https://www.sfu.ca/~ssurjano/camel6.html

"""
__all__ = [
    "six_hump_camel",
    "six_hump_camel_simple",
    "six_hump_camel_with_variable_resources",
    "six_hump_camel_CUDA_variable_resources",
    "persistent_six_hump_camel",
]

import os
import sys
import numpy as np
import time
from libensemble.executors.executor import Executor
from libensemble.message_numbers import UNSET_TAG, WORKER_DONE, TASK_FAILED
from libensemble.resources.resources import Resources
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, EVAL_SIM_TAG, FINISHED_PERSISTENT_SIM_TAG


def six_hump_camel(H, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a collection of points given in ``H['x']``.
    Additionally evaluates the gradient if ``'grad'`` is a field in
    ``sim_specs['out']`` and pauses for ``sim_specs['user']['pause_time']]`` if
    defined.

    .. seealso::
        `test_old_aposmm_with_gradients.py  <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_old_aposmm_with_gradients.py>`_ # noqa
    """

    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        H_o["f"][i] = six_hump_camel_func(x)

        if "grad" in H_o.dtype.names:
            H_o["grad"][i] = six_hump_camel_grad(x)

        if "user" in sim_specs and "pause_time" in sim_specs["user"]:
            time.sleep(sim_specs["user"]["pause_time"])

    return H_o, persis_info


def six_hump_camel_simple(x, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a single point ``x``.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    H_o = np.zeros(1, dtype=sim_specs["out"])

    H_o["f"] = six_hump_camel_func(x[0][0])

    if "pause_time" in sim_specs["user"]:
        time.sleep(sim_specs["user"]["pause_time"])

    return H_o, persis_info


def six_hump_camel_with_variable_resources(H, persis_info, sim_specs, libE_info):
    """
    Evaluates the six hump camel for a collection of points given in ``H['x']``
    via the executor, supporting variable sized simulations/resources, as
    determined by the generator.

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

        task = exctr.submit(
            app_name=app,
            app_args=inpt,
            num_procs=nprocs,
            stdout="out.txt",
            stderr="err.txt",
            dry_run=dry_run,
        )
        task.wait()
        # while(not task.finished):
        #     time.sleep(0.1)
        #     task.poll()

        task_states.append(task.state)

        if app == "six_hump_camel":
            # H_o['f'][i] = float(task.read_stdout())  # Reads whole file
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

    return H_o, persis_info, calc_status


def six_hump_camel_CUDA_variable_resources(H, persis_info, sim_specs, libE_info):
    """Launches an app setting GPU resources

    The standard test apps do not run on GPU, but demonstrates accessing resource
    information to set ``CUDA_VISIBLE_DEVICES``, and typical run configuration.
    """
    x = H["x"][0]
    H_o = np.zeros(1, dtype=sim_specs["out"])

    # Interrogate resources available to this worker
    resources = Resources.resources.worker_resources
    slots = resources.slots

    assert resources.matching_slots, f"Error: Cannot set CUDA_VISIBLE_DEVICES when unmatching slots on nodes {slots}"

    resources.set_env_to_slots("CUDA_VISIBLE_DEVICES")
    num_nodes = resources.local_node_count
    cores_per_node = resources.slot_count  # One CPU per GPU

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
    )

    task.wait()  # Wait for run to complete

    # Access app output
    with open("out.txt") as f:
        H_o["f"] = float(f.readline().strip())  # Read just first line

    calc_status = WORKER_DONE if task.state == "FINISHED" else "FAILED"
    return H_o, persis_info, calc_status


def persistent_six_hump_camel(H, persis_info, sim_specs, libE_info):
    """
    Similar to ``six_hump_camel``, but runs in persistent mode.
    """

    ps = PersistentSupport(libE_info, EVAL_SIM_TAG)

    # Either start with a work item to process - or just start and wait for data
    if H.size > 0:
        tag = None
        Work = None
        calc_in = H
    else:
        tag, Work, calc_in = ps.recv()

    while tag not in [STOP_TAG, PERSIS_STOP]:

        # calc_in: This should either be a function (unpack_work ?) or included/unpacked in ps.recv/ps.send_recv.
        if Work is not None:
            persis_info = Work.get("persis_info", persis_info)
            libE_info = Work.get("libE_info", libE_info)

        # Call standard six_hump_camel sim
        H_o, persis_info = six_hump_camel(calc_in, persis_info, sim_specs, libE_info)

        tag, Work, calc_in = ps.send_recv(H_o)

    final_return = None

    # Overwrite final point - for testing only
    if sim_specs["user"].get("replace_final_fields", 0):
        calc_in = np.ones(1, dtype=[("x", float, (2,))])
        H_o, persis_info = six_hump_camel(calc_in, persis_info, sim_specs, libE_info)
        final_return = H_o

    return final_return, persis_info, FINISHED_PERSISTENT_SIM_TAG


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    return term1 + term2 + term3


def six_hump_camel_grad(x):
    """
    Definition of the six-hump camel gradient
    """

    x1 = x[0]
    x2 = x[1]
    grad = np.zeros(2)

    grad[0] = 2.0 * (x1**5 - 4.2 * x1**3 + 4.0 * x1 + 0.5 * x2)
    grad[1] = x1 + 16 * x2**3 - 8 * x2

    return grad


if __name__ == "__main__":
    x = (float(sys.argv[1]), float(sys.argv[2]))
    result = six_hump_camel_func(x)
    print(result)
