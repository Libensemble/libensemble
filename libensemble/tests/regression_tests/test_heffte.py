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

from libensemble import Ensemble
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.heffte import call_and_process_heffte as sim_f
from libensemble.specs import AllocSpecs, ExitCriteria, SimSpecs

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

    study = Ensemble(parse_args=True)
    study.sim_specs = SimSpecs(
        sim_f=sim_f,
        inputs=["exec_and_args"],
        out=[("RUN_TIME", float)],
    )

    n_samp = len(full_list)
    H0 = np.zeros(n_samp, dtype=[("exec_and_args", str, max_len), ("sim_id", int), ("sim_started", bool)])
    H0["exec_and_args"] = full_list
    H0["sim_id"] = range(n_samp)

    study.alloc_specs = AllocSpecs(alloc_f=alloc_f)
    study.exit_criteria = ExitCriteria(sim_max=len(H0))
    study.H0 = H0
    study.run()

    if study.is_manager:
        assert len(study.H) == len(study.H0)
        assert np.all(study.H["sim_ended"]), "Every point should have been marked as ended"
        assert len(np.unique(study.H["RUN_TIME"])) >= len(study.H) / 2, "Most of the RUN_TIMEs should be unique"
        print("\nlibEnsemble correctly didn't add anything to initial sample")
        study.save_output(__file__)
