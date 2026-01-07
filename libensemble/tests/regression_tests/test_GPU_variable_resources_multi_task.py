"""
Tests multi-task (using GPU and non-GPU tasks using variable resources
and automatic GPU assignment in libEnsemble.

The persistent generator creates simulations with variable resource requirements
and some that require GPUs and some do not. The "num_procs" and "num_gpus"
for each task are set in the generator. These are automatically passed through
to the executor used by the sim.

The sim_f (gpu_variable_resources_from_gen) asserts that GPUs assignment
is correct for the default method for the MPI runner. GPUs are not actually
used for default application. CPUs and GPUs per node are mocked up below
(if this line is removed, libEnsemble will detect any CPUs/GPUs available).

A dry_run option is provided. This can be set in the calling script, and will
just print run-lines and GPU settings. This may be used for testing run-lines
produced and GPU settings for different MPI runners.

Execute via one of the following commands (e.g. 9 workers):
   mpiexec -np 10 python test_GPU_variable_resources_multi_task.py
   python test_GPU_variable_resources_multi_task.py --nworkers 9

When running with the above command, the number of concurrent evaluations of
the objective function will be 8, as one of the nine workers will be the
persistent generator.

This test must be run with 9 or more workers (8 sim workers), in order
to resource all works units. More generally:
    ((nworkers - 1) - gpus_on_node) >= gen_specs["user"][max_procs]

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 10

import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor

# Using num_procs / num_gpus in gen
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_diff_simulations as gen_f

# Import libEnsemble items for this test
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources_from_gen as sim_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
from libensemble.tools import add_unique_random_streams

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    # Get paths for applications to run
    six_hump_camel_app = six_hump_camel.__file__
    exctr = MPIExecutor()
    exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

    gpu_test = Ensemble(parse_args=True, executor=exctr)
    nworkers = gpu_test.nworkers
    gpu_test.libE_specs = LibeSpecs(
        num_resource_sets=gpu_test.nworkers - 1,
        resource_info={"cores_on_node": (32, 64), "gpus_on_node": 4},
        sim_dirs_make=True,
        ensemble_dir_path="./ensemble_GPU_variable_multi_task_w" + str(nworkers),
    )

    gpu_test.sim_specs = SimSpecs(
        sim_f=sim_f,
        inputs=["x"],
        out=[("f", float)],
        user={"dry_run": False},
    )
    gpu_test.gen_specs = GenSpecs(
        gen_f=gen_f,
        persis_in=["f", "x", "sim_id"],
        out=[("num_procs", int), ("num_gpus", int), ("x", float, 2)],
        user={
            "initial_batch_size": nworkers - 1,
            "max_procs": (nworkers - 1) // 2,  # Any sim created can req. 1 worker up to max
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "multi_task": True,
        },
    )

    gpu_test.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "give_all_with_same_priority": False,
            "async_return": False,  # False causes batch returns
        },
    )

    gpu_test.persis_info = add_unique_random_streams({}, gpu_test.nworkers + 1)
    gpu_test.exit_criteria = ExitCriteria(sim_max=40, wallclock_max=300)

    if gpu_test.ready():
        gpu_test.run()

    if gpu_test.is_manager:
        assert gpu_test.flag == 0
        gpu_test.save_output(__file__)
