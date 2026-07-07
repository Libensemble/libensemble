"""
VOCS-based version of test_2d_sampling.py. using the
 ``LatinHypercubeSample`` class.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_2d_sampling_vocs.py
   python test_2d_sampling_vocs.py --nworkers 3
   python test_2d_sampling_vocs.py --nworkers 3 --comms tcp
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local threads tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np
from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes.sampling import LatinHypercubeSample
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


def sim_f(In, persis_info, sim_specs, _):
    Out = np.zeros(1, dtype=sim_specs["out"])
    Out["f"] = np.sqrt(In["x0"] ** 2 + In["x1"] ** 2)
    return Out, persis_info


if __name__ == "__main__":
    sampling = Ensemble(parse_args=True)
    sampling.libE_specs = LibeSpecs(save_every_k_sims=100)
    sampling.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x0", "x1"], outputs=[("f", float)])

    vocs = VOCS(
        variables={"x0": [-3.0, 3.0], "x1": [-2.0, 2.0]},
        objectives={"f": "MINIMIZE"},
    )
    generator = LatinHypercubeSample(vocs, random_seed=1)

    sampling.gen_specs = GenSpecs(
        generator=generator,
        persis_in=["x0", "x1", "f", "sim_id"],
        outputs=[("x0", float), ("x1", float)],
        initial_batch_size=100,
        batch_size=100,
    )

    sampling.exit_criteria = ExitCriteria(sim_max=200)

    sampling.run()
    if sampling.is_manager:
        assert len(sampling.H) >= 200
        x0 = sampling.H["x0"]
        x1 = sampling.H["x1"]
        f = sampling.H["f"]
        assert np.all(np.isclose(f, np.sqrt(x0 ** 2 + x1 ** 2)))
        print("\nlibEnsemble has calculated the 2D vector norm of all points")
    sampling.save_output(__file__)
