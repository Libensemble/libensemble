"""
Runs libEnsemble with uniform random sampling and writes results into sim dirs.
Tests sim_input_dir capabilities

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_sim_input_dir_option.py
   python test_sim_input_dir_option.py --nworkers 3
   python test_sim_input_dir_option.py --nworkers 3 --comms tcp

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
    sim_input_dir = os.path.abspath("./sim_input_dir")
    dir_to_copy = sim_input_dir + "/copy_this"

    for dire in [sim_input_dir, dir_to_copy]:
        if not os.path.isdir(dire):
            os.makedirs(dire, exist_ok=True)

    libE_specs["sim_input_dir"] = sim_input_dir
    libE_specs["sim_dirs_make"] = False
    libE_specs["sim_dir_symlink_files"] = [
        os.path.abspath("./test_sim_input_dir_option.py")
    ]  # to cover FileExistsError catch
    libE_specs["ensemble_copy_back"] = True
    libE_specs["use_workflow_dir"] = True

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

    ensemble_lens = []
    stats_lens = []

    for i in range(2):
        libE_specs["workflow_dir_path"] = (
            "./test_workflow" + str(i) + "_nworkers" + str(nworkers) + "_comms-" + libE_specs["comms"]
        )

        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

        assert os.path.isdir(libE_specs["workflow_dir_path"]), "workflow_dir not created"
        assert all(
            [
                i in os.listdir(libE_specs["workflow_dir_path"])
                for i in ["ensemble.log", "libE_stats.txt", "ensemble", "ensemble_back"]
            ]
        )

        with open(os.path.join(libE_specs["workflow_dir_path"], "ensemble.log"), "r") as f:
            lines = f.readlines()
            ensemble_lens.append(len(lines))

        with open(os.path.join(libE_specs["workflow_dir_path"], "libE_stats.txt"), "r") as f:
            lines = f.readlines()
            stats_lens.append(len(lines))

    assert ensemble_lens[0] == ensemble_lens[1], "ensemble.log's didn't have same length"
    assert stats_lens[0] == stats_lens[1], "libE_stats.txt's didn't have same length"
