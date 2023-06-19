"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling.py
   python test_1d_sampling.py --nworkers 3 --comms local
   python test_1d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

from libensemble import Ensemble
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.one_d_func import one_d_example as sim_f
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    sampling = Ensemble()
    sampling.libE_specs = LibeSpecs(save_every_k_gens=300, safe_mode=False, disable_log_files=True)
    sampling.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], out=[("f", float)])
    sampling.gen_specs = GenSpecs(
        gen_f=gen_f,
        out=[("x", float, (1,))],
        user={
            "gen_batch_size": 500,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    )

    sampling.add_random_streams()
    sampling.exit_criteria = ExitCriteria(gen_max=501)

    sampling.run()
    if sampling.is_manager:
        assert len(sampling.H) >= 501
        print("\nlibEnsemble with random sampling has generated enough points")
        sampling.save_output(__file__)
