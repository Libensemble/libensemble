"""
Runs libEnsemble with uniform random sampling and writes results into sim
dirs. This tests per-calculation sim_dir capabilities

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_sim_dirs_per_calc.py
   python test_sim_dirs_per_calc.py --nworkers 3
   python test_sim_dirs_per_calc.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import os

import numpy as np

from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import write_sim_func as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

nworkers, is_manager, libE_specs, _ = parse_args()

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    sim_input_dir = "./sim_input_dir"
    dir_to_copy = sim_input_dir + "/copy_this"
    dir_to_symlink = sim_input_dir + "/symlink_this"
    c_ensemble = "./ensemble_calcdirs_w" + str(nworkers) + "_" + libE_specs.get("comms")
    print("creating ensemble dir: ", c_ensemble, flush=True)

    for dir in [sim_input_dir, dir_to_copy, dir_to_symlink]:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    libE_specs["sim_dirs_make"] = True
    libE_specs["ensemble_dir_path"] = c_ensemble
    libE_specs["use_worker_dirs"] = False
    libE_specs["sim_dir_copy_files"] = [dir_to_copy]
    libE_specs["sim_dir_symlink_files"] = [dir_to_symlink]
    libE_specs["ensemble_copy_back"] = True
    libE_specs["reuse_output_dir"] = True

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "out": [("x", float, (1,))],
        "user": {
            "gen_batch_size": 20,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 21}

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert os.path.isdir(c_ensemble), f"Ensemble directory {c_ensemble} not created."
        dir_sum = sum(["sim" in i for i in os.listdir(c_ensemble)])
        assert (
            dir_sum == exit_criteria["sim_max"]
        ), f"Number of sim directories ({dir_sum}) does not match sim_max ({exit_criteria['sim_max']})."

        input_copied = []

        for base, files, _ in os.walk(c_ensemble):
            basedir = base.split("/")[-1]
            if basedir.startswith("sim"):
                input_copied.append(
                    all(
                        [
                            os.path.basename(j) in files
                            for j in libE_specs["sim_dir_copy_files"] + libE_specs["sim_dir_symlink_files"]
                        ]
                    )
                )

        assert all(input_copied), "Exact input files not copied or symlinked to each calculation directory"
