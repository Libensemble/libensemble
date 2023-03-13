"""
Test libEnsemble's integration with calling the heFFTe executable with various
configurations.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import itertools
from os.path import exists

import numpy as np

from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.heffte import call_and_process_heffte as sim_f
from libensemble.tools import parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    assert exists("speed3d_c2c"), "The heFFTe executable doesn't exist"

    fixed = ["mpirun -np 4 ./speed3d_c2c fftw double 128 128 128"]
    arg1 = ["-no-reorder", "-reorder"]
    arg2 = ["-a2a", "-a2av", "-p2p", "-p2p_pl"]
    arg3 = ["-ingrid 4 1 1", "-ingrid 2 2 1"]
    arg4 = ["-outgrid 4 1 1", "-outgrid 2 2 1"]

    part_list = list(itertools.product(fixed, arg1, arg2, arg3, arg4))

    full_list = list(map(" ".join, part_list))

    max_len = max([len(item) for item in full_list])

    nworkers, is_manager, libE_specs, _ = parse_args()

    sim_specs = {
        "sim_f": sim_f,
        "in": ["exec_and_args"],
        "out": [("RUN_TIME", float)],
    }

    gen_specs = {}

    n_samp = len(full_list)

    H0 = np.zeros(n_samp, dtype=[("exec_and_args", str, max_len), ("sim_id", int), ("sim_started", bool)])

    H0["exec_and_args"] = full_list
    H0["sim_id"] = range(n_samp)

    alloc_specs = {"alloc_f": alloc_f}

    exit_criteria = {"sim_max": len(H0)}

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, libE_specs=libE_specs, H0=H0
    )

    if is_manager:
        assert len(H) == len(H0)
        assert np.all(H["sim_ended"]), "Every point should have been marked as ended"
        assert len(np.unique(H["RUN_TIME"])) >= len(H) / 2, "Most of the RUN_TIMEs should be unique"
        print("\nlibEnsemble correctly didn't add anything to initial sample")
        save_libE_output(H, persis_info, __file__, nworkers)
