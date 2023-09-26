"""
Runs libEnsemble with the fd_param_finder persistent gen_f, which finds an
appropriate finite-difference parameter for the sim_f mapping from R^n to R^p
around the point x.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_fd_param_finder.py

When running with the above command, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_OS_SKIP: OSX
# TESTSUITE_EXTRA: true

import shutil  # For ECnoise.m

import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.start_fd_persistent import finite_diff_alloc as alloc_f
from libensemble.gen_funcs.persistent_fd_param_finder import fd_param_finder as gen_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.noisy_vector_mapping import func_wrapper as sim_f
from libensemble.sim_funcs.noisy_vector_mapping import noisy_function
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, SimSpecs
from libensemble.tools import add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    x0 = np.array([1.23, 4.56])  # point about which we are calculating finite difference parameters
    f0 = noisy_function(x0)
    n = len(x0)
    p = len(f0)

    fd_test = Ensemble(
        parse_args=True,
        sim_specs=SimSpecs(
            sim_f=sim_f,
            inputs=["x", "f_ind"],
            out=[("f_val", float)],
        ),
        gen_specs=GenSpecs(
            gen_f=gen_f,
            persis_in=["x", "f_val", "n_ind", "f_ind", "x_ind", "sim_id"],
            out=[("x", float, (n,)), ("n_ind", int), ("f_ind", int), ("x_ind", int)],
            user={
                "x0": x0,
                "f0": f0,
                "nf": 10,
                "p": p,
                "n": n,
                "noise_h_mat": np.multiply(np.logspace(-16, -1, p), np.ones((n, p))),
                "maxnoiseits": 3,
            },
        ),
        alloc_specs=AllocSpecs(alloc_f=alloc_f),
        exit_criteria=ExitCriteria(gen_max=1000),
    )
    fd_test.persis_info = add_unique_random_streams({}, fd_test.nworkers + 1)

    shutil.copy("./scripts_used_by_reg_tests/ECnoise.m", "./")

    H, persis_info, _ = fd_test.run()

    if fd_test.is_manager:
        assert len(H) < fd_test.exit_criteria.gen_max, "Problem didn't stop early, which should have been the case."
        assert np.all(persis_info[1]["Fnoise"] > 0), "gen_f didn't find noise for all F_i components."

        fd_test.save_output(__file__)
