"""Persistent random sampling using various methods of dynamic resource assignment

Each function generates points uniformly over the domain defined by ``gen_specs["user"]["ub"]``
and ``gen_specs["user"]["lb"]``.

Most functions use a random request of resources over a range, setting num_procs, num_gpus, or
resource sets. The function ``uniform_sample_with_var_gpus`` uses the ``x`` value to determine
the number of GPUs requested.
"""

import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.tools.test_support import check_gpu_setting

__all__ = [
    "uniform_sample",
    "uniform_sample_with_var_gpus",
    "uniform_sample_with_procs_gpus",
    "uniform_sample_with_var_priorities",
    "uniform_sample_diff_simulations",
    "uniform_sample_with_sim_gen_resources",
]


def _get_user_params(user_specs):
    """Extract user params"""
    b = user_specs["initial_batch_size"]
    ub = user_specs["ub"]
    lb = user_specs["lb"]
    n = len(lb)  # dimension
    return b, n, lb, ub


def uniform_sample(_, persis_info, gen_specs, libE_info):
    """
    Randomly requests a different number of resource sets to be used in the
    evaluation of the generated points.

    .. seealso::
        `test_uniform_sampling_with_variable_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling_with_variable_resources.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    rng = persis_info["rand_stream"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None

    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = rng.uniform(lb, ub, (b, n))
        H_o["resource_sets"] = rng.integers(1, gen_specs["user"]["max_resource_sets"] + 1, b)
        print(f"GEN created {b} sims, with resource sets req. of size(s) {H_o['resource_sets']}", flush=True)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def uniform_sample_with_var_gpus(_, persis_info, gen_specs, libE_info):
    """
    Requests a number of GPUs based on the ``x`` value to be used in the evaluation
    of the generated points. By default, simulations will assign one MPI processor
    per GPU.

    Note that the ``num_gpus`` gen_specs["out"] option (similar to ``num_procs``) does
    not need to be passed as a sim_specs["in"]. It will automatically be passed to
    simulation functions and used by any MPI Executor unless overridden in the
    ``executor.submit`` function.

    .. seealso::
        `test_GPU_variable_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_GPU_variable_resources.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    rng = persis_info["rand_stream"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None
    max_gpus = gen_specs["user"]["max_gpus"]

    while tag not in [STOP_TAG, PERSIS_STOP]:
        x = rng.uniform(lb, ub, (b, n))
        bucket_size = (ub[0] - lb[0]) / max_gpus

        # Determine number of GPUs based on linear split over x range (first dimension).
        ngpus = [int((num - lb[0]) / bucket_size) + 1 for num in x[:, 0]]

        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = x
        H_o["num_gpus"] = ngpus

        print(f"GEN created {b} sims requiring {ngpus} GPUs", flush=True)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def uniform_sample_with_procs_gpus(_, persis_info, gen_specs, libE_info):
    """
    Randomly requests a different number of processors and gpus to be used in the
    evaluation of the generated points.

    .. seealso::
        `test_GPU_variable_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_GPU_variable_resources.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    rng = persis_info["rand_stream"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None

    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = rng.uniform(lb, ub, (b, n))
        nprocs = rng.integers(1, gen_specs["user"]["max_procs"] + 1, b)
        H_o["num_procs"] = nprocs  # This would get matched to GPUs anyway, if no other config given
        H_o["num_gpus"] = nprocs
        print(f"GEN created {b} sims requiring {nprocs} procs. One GPU per proc", flush=True)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def uniform_sample_with_var_priorities(_, persis_info, gen_specs, libE_info):
    """
    Initial batch has matching priorities, after which a different number of
    resource sets and priorities are requested for each point.
    """

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    rng = persis_info["rand_stream"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    H_o = np.zeros(b, dtype=gen_specs["out"])
    H_o["x"] = rng.uniform(lb, ub, (b, n))
    H_o["resource_sets"] = 1
    H_o["priority"] = 1

    print(f"GEN created {b} sims, with resource sets req. of size(s) {H_o['resource_sets']}", flush=True)

    # Send batches until manager sends stop tag
    tag, Work, calc_in = ps.send_recv(H_o)

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)  # no. of points returned

        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = rng.uniform(lb, ub, (b, n))
        H_o["resource_sets"] = rng.integers(1, gen_specs["user"]["max_resource_sets"] + 1, b)
        H_o["priority"] = 10 * H_o["resource_sets"]
        print(f"GEN created {b} sims, with resource sets req. of size(s) {H_o['resource_sets']}", flush=True)

        tag, Work, calc_in = ps.send_recv(H_o)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def uniform_sample_diff_simulations(_, persis_info, gen_specs, libE_info):
    """
    Randomly requests a different number of processors for each simulation.
    One simulation type also uses GPUs.

    .. seealso::
        `test_GPU_variable_resources_multi_task.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_GPU_variable_resources_multi_task.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    rng = persis_info["rand_stream"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None

    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = rng.uniform(lb, ub, (b, n))

        nprocs = rng.integers(1, gen_specs["user"]["max_procs"] + 1, b)
        use_gpus = rng.choice([True, False], b)
        H_o["num_procs"] = nprocs
        H_o["num_gpus"] = np.where(use_gpus, nprocs, 0)
        if "app_type" in H_o.dtype.names:
            H_o["app_type"] = np.where(use_gpus, "gpu_app", "cpu_app")

        print(f"\nGEN created {b} sims requiring {nprocs} procs. Use GPUs {use_gpus}", flush=True)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def uniform_sample_with_sim_gen_resources(_, persis_info, gen_specs, libE_info):
    """
    Randomly requests a different number of processors and gpus to be used in the
    evaluation of the generated points.

    .. seealso::
        `test_GPU_variable_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_GPU_variable_resources.py>`_
    """  # noqa

    b, n, lb, ub = _get_user_params(gen_specs["user"])
    rng = persis_info["rand_stream"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None

    dry_run = gen_specs["user"].get("dry_run", False)  # logs run lines instead of running

    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs["out"])
        H_o["x"] = rng.uniform(lb, ub, (b, n))

        # Run an app using resources given by libE_specs or persis_info (test purposes only)
        task = Executor.executor.submit(
            app_name="six_hump_camel",
            app_args="-0.99 -0.19",
            stdout="out.txt",
            stderr="err.txt",
            dry_run=dry_run,
        )

        if not dry_run:
            task.wait()  # Wait for run to complete

        # Asserts GPU set correctly (for known MPI runners)
        check_gpu_setting(task, print_setting=True)

        # Set resources for sims
        nprocs = rng.integers(1, gen_specs["user"]["max_procs"] + 1, b)
        H_o["num_procs"] = nprocs  # This would get matched to GPUs anyway, if no other config given
        H_o["num_gpus"] = nprocs
        print(f"GEN created {b} sims requiring {nprocs} procs. One GPU per proc", flush=True)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
