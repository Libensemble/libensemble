"""
Tests libEnsemble's capability to take in an existing sample of points with
sim_f values and do additional evaluations.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_evaluate_mixed_sample.py
   python test_evaluate_mixed_sample.py --nworkers 3
   python test_evaluate_mixed_sample.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import warnings

import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.borehole import borehole as sim_f
from libensemble.sim_funcs.borehole import borehole_func, gen_borehole_input
from libensemble.specs import AllocSpecs, ExitCriteria, SimSpecs

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    samp = 1000
    n = 8

    H0 = np.zeros(
        samp, dtype=[("x", float, n), ("f", float), ("sim_id", int), ("sim_started", bool), ("sim_ended", bool)]
    )
    np.random.seed(0)
    H0["x"] = gen_borehole_input(samp)
    for i in range(500):
        H0["f"][i] = borehole_func(H0["x"][i])
    H0["sim_started"][:500] = True
    H0["sim_ended"][:500] = True

    sampling = Ensemble(parse_args=True)
    sampling.libE_specs.gen_on_manager = True
    sampling.H0 = H0
    sampling.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], out=[("f", float)])
    sampling.alloc_specs = AllocSpecs(alloc_f=alloc_f)
    sampling.exit_criteria = ExitCriteria(sim_max=len(H0))
    sampling.run()

    if sampling.is_manager:
        assert len(sampling.H) == len(H0)
        assert np.array_equal(H0["x"], sampling.H["x"])
        assert np.all(sampling.H["sim_ended"])
        print("\nlibEnsemble correctly didn't add anything to initial sample")
        sampling.save_output(__file__)
