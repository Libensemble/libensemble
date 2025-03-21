"""
Runs libEnsemble with Latin hypercube sampling on a simple 2D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_2d_sampling.py
   python test_2d_sampling.py --nworkers 3
   python test_2d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local threads tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

from libensemble import Ensemble
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.simple_sim import norm_eval as sim_f
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    sampling = Ensemble(parse_args=True)
    sampling.libE_specs = LibeSpecs(save_every_k_sims=100)
    sampling.sim_specs = SimSpecs(sim_f=sim_f)
    sampling.gen_specs = GenSpecs(
        gen_f=gen_f,
        outputs=[("x", float, 2)],
        user={
            "gen_batch_size": 100,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    )

    sampling.exit_criteria = ExitCriteria(sim_max=200)
    sampling.add_random_streams()

    sampling.run()
    if sampling.is_manager:
        assert len(sampling.H) >= 200
        x = sampling.H["x"]
        f = sampling.H["f"]
        assert np.all(np.isclose(f, np.sqrt(np.sum(x**2, axis=1))))
        print("\nlibEnsemble has calculated the 2D vector norm of all points")
    sampling.save_output(__file__)
