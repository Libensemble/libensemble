"""
Runs libEnsemble with APOSMM with a PETSc/TAO local optimizer and using
the executor to run an application.

This is to test the scenario, where Open-MPI will fail due to nested MPI, if
PETSc is imported at global level.

Execute via one of the following commands (e.g., 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_tao_nm.py
   python test_with_app_persistent_aposmm_tao_nm.py --nworkers 3
   python test_with_app_persistent_aposmm_tao_nm.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, since one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
import sys

import numpy as np

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import multi_points_with_variable_resources as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

# For Open-MPI the following lines cannot be used, thus allowing PETSc to import.
# import libensemble.gen_funcs
# libensemble.gen_funcs.rc.aposmm_optimizers = "petsc"


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()
    exctr = MPIExecutor()
    six_hump_camel_app = six_hump_camel.__file__
    exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")
    libE_specs["sim_dirs_make"] = True

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("grad", float, n)],
        "user": {"app": "six_hump_camel"},  # helloworld or six_hump_camel
    }

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "grad"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 10,
            "localopt_method": "nm",
            "lb": np.array([-3, -2]),  # This is only for sampling. TAO_NM doesn't honor constraints.
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 20}  # must be bigger than sample size to enter into optimization code.

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
