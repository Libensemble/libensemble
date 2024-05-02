"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem
Unlike other tests, the command line options are not parsed.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling_no_comms_given.py
   python test_1d_sampling_no_comms_given.py

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# Note for this test: NPROCS on command line will be ignored for local comms
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4

import numpy as np

from libensemble import Ensemble
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.one_d_func import one_d_example as sim_f
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
from libensemble.tools import add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    sampling = Ensemble(parse_args=False)
    sampling.libE_specs = LibeSpecs(
        save_every_k_gens=300,
        safe_mode=False,
        disable_log_files=True,
        nworkers=3
    )
    sampling.sim_specs = SimSpecs(
        sim_f=sim_f,
        inputs=["x"],
        outputs=[("f", float)],
    )
    sampling.gen_specs = GenSpecs(
        gen_f=gen_f,
        outputs=[("x", float, (1,))],
        user={
            "gen_batch_size": 500,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    )

    sampling.persis_info = add_unique_random_streams({}, sampling.nworkers + 1)
    sampling.exit_criteria = ExitCriteria(gen_max=501)

    sampling.run()
    if sampling.is_manager:
        assert len(sampling.H) >= 501
        print("\nlibEnsemble with random sampling has generated enough points")
        sampling.save_output(__file__)
