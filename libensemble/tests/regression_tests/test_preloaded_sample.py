"""
Evaluate a pre-existing sample through PreloadedSampleGenerator.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_preloaded_sample.py
   python test_preloaded_sample.py --nworkers 3
   python test_preloaded_sample.py --nworkers 3 --comms threads
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local threads
# TESTSUITE_NPROCS: 4

import numpy as np
from gest_api.vocs import VOCS, ContinuousVariable

from libensemble import Ensemble
from libensemble.gen_classes.preloaded import PreloadedSampleGenerator
from libensemble.sim_funcs.borehole import borehole as sim_f
from libensemble.sim_funcs.borehole import gen_borehole_input
from libensemble.specs import GenSpecs, SimSpecs

if __name__ == "__main__":
    n_samp = 20
    points = np.zeros(n_samp, dtype=[("x", float, 8), ("sim_id", int), ("sim_started", bool)])
    np.random.seed(0)
    points["x"] = gen_borehole_input(n_samp)
    points["sim_id"] = range(n_samp)

    vocs = VOCS(
        variables={"x": ContinuousVariable(dtype=(float, (8,)), domain=[0.0, 1.0])},
        objectives={"f": "MINIMIZE"},
    )
    sampling = Ensemble(parse_args=True)
    sampling.gen_specs = GenSpecs(
        generator=PreloadedSampleGenerator(points, vocs=vocs),
        vocs=vocs,
    )
    sampling.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], out=[("f", float)])
    sampling.run(sim_max=n_samp)

    if sampling.is_manager:
        assert len(sampling.H) == n_samp
        np.testing.assert_array_equal(sampling.H["x"], points["x"])
        assert np.all(sampling.H["sim_ended"])
        assert np.all(np.isfinite(sampling.H["f"]))
        sampling.save_output(__file__)
