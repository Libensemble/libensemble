import glob
import os
import re

import numpy as np

# Import libEnsemble items for this test
from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample as gen_f
from libensemble.resources.platforms import PerlmutterGPU
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources as sim_f
from libensemble.specs import LibeSpecs

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    # Get paths for applications to run
    six_hump_camel_app = six_hump_camel.__file__
    n = 2

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "give_all_with_same_priority": False,
            "async_return": False,  # False batch returns
        },
    }

    exit_criteria = {"sim_max": 20}

    # Ensure LIBE_PLATFORM environment variable is not set.
    if "LIBE_PLATFORM" in os.environ:
        del os.environ["LIBE_PLATFORM"]

    exctr = MPIExecutor()
    exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

    ensemble = Ensemble(
        parse_args=True,
        executor=exctr,
        alloc_specs=alloc_specs,
        exit_criteria=exit_criteria,
        # libE_specs = LibeSpecs(use_workflow_dir=True, platform_specs=platform_specs),  # works
    )

    platform_specs = PerlmutterGPU()
    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=ensemble.nworkers - 1,
        resource_info={"gpus_on_node": 4},
        use_workflow_dir=True,
        platform_specs=platform_specs,
    )

    ensemble.gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "x", "sim_id"],
        "out": [("priority", float), ("resource_sets", int), ("x", float, n)],
        "user": {
            "initial_batch_size": ensemble.nworkers - 1,
            "max_resource_sets": ensemble.nworkers - 1,  # Any sim created can req. 1 worker up to all.
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    ensemble.sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"dry_run": True},
    }

    ensemble.add_random_streams()
    ensemble.run()

    if ensemble.is_manager:
        matching_dirs = glob.glob("workflow_*")
        assert matching_dirs, "No workflow dir found"
        most_recent_dir = max(matching_dirs, key=os.path.getctime)
        print(f"Checking ensemble.log in {most_recent_dir}")
        file_path = file_path = os.path.join(most_recent_dir, "ensemble.log")
        if os.path.exists(file_path):  # an assert
            with open(file_path, "r") as file:
                content = file.read()
                pattern = r"Runline:\s+srun"
                assert re.findall(pattern, content), "Incorrect MPI runner"
