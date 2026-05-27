"""
Test libEnsemble's capability to evaluate existing points and then generate
new samples.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_evaluate_existing_sample.py
   python test_evaluate_existing_sample.py --nworkers 3
   python test_evaluate_existing_sample.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np
from gest_api.vocs import VOCS

# Import libEnsemble items for this test
from libensemble import Ensemble
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, SimSpecs


def create_H0(lb, ub, H0_size):
    """Create an H0 for give_pregenerated_sim_work"""
    n = len(lb)
    b = H0_size

    H0 = np.zeros(b, dtype=[("x", float, 2), ("sim_id", int), ("sim_started", bool)])
    H0["x"] = np.random.uniform(lb, ub, (b, n))
    H0["sim_id"] = range(b)
    H0["sim_started"] = False
    return H0


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    sampling = Ensemble(parse_args=True)
    sampling.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], out=[("f", float)])

    vocs = VOCS(variables={"x0": [-3, 3], "x1": [-3, 3]}, objectives={"f": "MINIMIZE"})
    lb = np.array([-3, -3])
    ub = np.array([3, 3])

    sampling.gen_specs = GenSpecs(
        gen_f=gen_f,
        outputs=[("x", float, (2,))],
        batch_size=50,
        vocs=vocs,
    )
    sampling.exit_criteria = ExitCriteria(sim_max=100)
    sampling.H0 = create_H0(lb, ub, 50)
    sampling.alloc_specs = AllocSpecs(alloc_f=give_sim_work_first)
    sampling.run()

    if sampling.is_manager:
        assert len(sampling.H) == 2 * len(sampling.H0)
        assert np.array_equal(sampling.H0["x"][:50], sampling.H["x"][:50])
        assert np.all(sampling.H["sim_ended"])
        assert np.all(sampling.H["gen_worker"] == 0)
        print("\nlibEnsemble correctly appended to the initial sample via an additional gen.")
        sampling.save_output(__file__)
