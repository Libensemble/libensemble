# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python test_6-hump_camel_aposmm_LD_MMA.py
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4

import sys
import numpy as np
from math import gamma, pi, sqrt
from copy import deepcopy

# Import libEnsemble items for this test
from libensemble.libE import libE, libE_tcp_worker
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = ["nlopt", "petsc"]
from libensemble.gen_funcs.old_aposmm import aposmm_logic as gen_f

from libensemble.alloc_funcs.fast_alloc_to_aposmm import give_sim_work_first as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import (
    persis_info_1 as persis_info,
    aposmm_gen_out as gen_out,
    six_hump_camel_minima as minima,
)

nworkers, is_manager, libE_specs, _ = parse_args()

n = 2
sim_specs = {
    "sim_f": sim_f,
    "in": ["x"],
    "out": [("f", float), ("grad", float, n)],
}

gen_out += [("x", float, n), ("x_on_cube", float, n)]
gen_specs = {
    "gen_f": gen_f,
    "in": [o[0] for o in gen_out] + ["f", "grad", "sim_ended"],
    "out": gen_out,
    "user": {
        "initial_sample_size": 100,
        "sample_points": np.round(minima, 1),
        "localopt_method": "LD_MMA",
        "rk_const": 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
        "xtol_rel": 1e-3,
        "max_active_runs": 6,
        "lb": np.array([-3, -2]),
        "ub": np.array([3, 2]),
    },
}

alloc_specs = {
    "alloc_f": alloc_f,
    "out": [],
    "user": {"batch_mode": True, "num_active_gens": 1},
}

persis_info = add_unique_random_streams(persis_info, nworkers + 1)
persis_info_safe = deepcopy(persis_info)

exit_criteria = {"sim_max": 1000}

# Set up appropriate abort mechanism depending on comms
libE_abort = quit
if libE_specs["comms"] == "mpi":
    from mpi4py import MPI

    def libE_mpi_abort():
        MPI.COMM_WORLD.Abort(1)

    libE_abort = libE_mpi_abort

# Perform the run (TCP worker mode)
if libE_specs["comms"] == "tcp" and not is_manager:
    run = int(sys.argv[-1])
    libE_tcp_worker(sim_specs, gen_specs[run], libE_specs)
    quit()

# Perform the run
for run in range(3):
    if libE_specs["comms"] == "tcp" and is_manager:
        libE_specs["worker_cmd"].append(str(run))

    if run == 1:
        gen_specs["user"]["localopt_method"] = "blmvm"
        gen_specs["user"]["grtol"] = 1e-5
        gen_specs["user"]["gatol"] = 1e-5
        persis_info = deepcopy(persis_info_safe)

    if run == 2:
        gen_specs["user"]["localopt_method"] = "LD_MMA"
        # Change the bounds to put a local min at a corner point (to test that
        # APOSMM handles the same point being in multiple runs) ability to
        # give back a previously evaluated point)
        gen_specs["user"]["ub"] = np.array([-2.9, -1.9])
        gen_specs["user"]["mu"] = 1e-4
        gen_specs["user"]["rk_const"] = 0.01 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi)
        gen_specs["user"]["lhs_divisions"] = 2
        # APOSMM can be called when some run is incomplete
        alloc_specs["user"].pop("batch_mode")

        gen_specs["user"].pop("xtol_rel")
        gen_specs["user"]["ftol_rel"] = 1e-2
        gen_specs["user"]["xtol_abs"] = 1e-3
        gen_specs["user"]["ftol_abs"] = 1e-8
        exit_criteria = {"sim_max": 200}
        minima = np.array([[-2.9, -1.9]])

        persis_info = deepcopy(persis_info_safe)

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        if flag != 0:
            print(f"Exit was not on convergence (code {flag})", flush=True)
            libE_abort()

        tol = 1e-5
        for m in minima:
            # The minima are known on this test problem.
            # 1) We use their values to test APOSMM has identified all minima
            # 2) We use their approximate values to ensure APOSMM evaluates a
            #    point in each minima's basin of attraction.
            print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
            if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) > tol:
                libE_abort()

        print(
            "\nlibEnsemble with APOSMM using a gradient-based localopt method has identified the "
            + str(np.shape(minima)[0])
            + " minima within a tolerance "
            + str(tol)
        )
        save_libE_output(H, persis_info, __file__, nworkers)
