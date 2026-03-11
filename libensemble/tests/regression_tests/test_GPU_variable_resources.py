"""
Tests variable resource detection and automatic GPU assignment in libEnsemble

The persistent generator creates simulations with variable resource requirements.

The sim_f (gpu_variable_resources_from_gen) asserts that GPUs assignment
is correct for the default method for the MPI runner. GPUs are not actually
used for default application. Four GPUs per node is mocked up below (if this line
is removed, libEnsemble will detect any GPUs available).

A dry_run option is provided. This can be set in the calling script, and will
just print run-lines and GPU settings. This may be used for testing run-lines
produced and GPU settings for different MPI runners.

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 6 python test_GPU_variable_resources.py
   python test_GPU_variable_resources.py --nworkers 5

When running with the above command, the number of concurrent evaluations of
the objective function will be 4, as one of the five workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 6

import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_with_procs_gpus as gen_f1
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_with_var_gpus as gen_f2

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
    gpu_test.libE_specs = LibeSpecs(
        num_resource_sets=gpu_test.nworkers - 1,
        resource_info={"cores_on_node": (8, 16), "gpus_on_node": 4},
        sim_dirs_make=True,
        ensemble_dir_path="./ensemble_GPU_variable_w" + str(gpu_test.nworkers),
        reuse_output_dir=True,
    )

    gpu_test.sim_specs = SimSpecs(
        sim_f=sim_f,
        inputs=["x"],
        out=[("f", float)],
        user={"dry_run": False},
    )
    gpu_test.gen_specs = GenSpecs(
        gen_f=gen_f1,
        persis_in=["f", "x", "sim_id"],
        out=[("num_procs", int), ("num_gpus", int), ("x", float, 2)],
        user={
            "initial_batch_size": gpu_test.nworkers - 1,
            "max_procs": gpu_test.nworkers - 1,  # Any sim created can req. 1 worker up to max
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    )

    gpu_test.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "give_all_with_same_priority": False,
            "async_return": False,  # False causes batch returns
        },
    )

    # Run with random num_procs/num_gpus for each simulation
    gpu_test.persis_info = add_unique_random_streams({}, gpu_test.nworkers + 1)
    gpu_test.exit_criteria = ExitCriteria(sim_max=20)

    gpu_test.run()
    if gpu_test.is_manager:
        assert gpu_test.flag == 0

    # Run with num_gpus based on x[0] for each simulation
    gpu_test.gen_specs.gen_f = gen_f2
    gpu_test.gen_specs.user["max_gpus"] = gpu_test.nworkers - 1
    gpu_test.persis_info = add_unique_random_streams({}, gpu_test.nworkers + 1)
    gpu_test.exit_criteria = ExitCriteria(sim_max=20)
    gpu_test.run()

    if gpu_test.is_manager:
        assert gpu_test.flag == 0

        gpu_test.save_output(__file__)
