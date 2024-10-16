# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 8

import shutil

import numpy as np

from libensemble import Ensemble

# Import libEnsemble items for this test
from libensemble.executors import Executor
from libensemble.gen_funcs.sampling import latin_hypercube_sample
from libensemble.sim_funcs.effis_t3d import effis_t3d
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
from libensemble.tools import add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    sampling = Ensemble(parse_args=True)
    sampling.sim_specs = SimSpecs(sim_f=effis_t3d, inputs=["x"], outputs=[("Wtot_MJ", float)])
    sampling.libE_specs = LibeSpecs(sim_dirs_make=True)
    sampling.gen_specs = GenSpecs(
        gen_f=latin_hypercube_sample,
        outputs=[("x", float, (2,))],
        user={
            "gen_batch_size": 10,
            "lb": np.array([0, 0]),
            "ub": np.array([10, 10]),
        },
    )

    exctr = Executor()
    exctr.register_app(shutil.which("effis-submit"), app_name="effis")

    sampling.persis_info = add_unique_random_streams({}, sampling.nworkers + 1)
    sampling.exit_criteria = ExitCriteria(sim_max=10)

    sampling.run()
    if sampling.is_manager:
        assert len(sampling.H) >= 10
        print("\nlibEnsemble with random sampling has generated enough points")
    sampling.save_output(__file__)
