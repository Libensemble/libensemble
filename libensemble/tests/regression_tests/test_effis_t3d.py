# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 8

import shutil

import numpy as np

from libensemble import Ensemble

# Import libEnsemble items for this test
from libensemble.executors import Executor
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f
from libensemble.sim_funcs.effis_t3d import effis_t3d
from libensemble.specs import ExitCriteria, AllocSpecs, LibeSpecs, SimSpecs

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    core = np.linspace(0.34, 0.36, 11)
    edge = np.linspace(0.28, 0.3, 11)

    # Create the meshgrid
    X, Y = np.meshgrid(core, edge)

    pairs = np.array(list(zip(X.ravel(), Y.ravel())))
    n_samp = len(pairs)

    H0 = np.zeros(n_samp, dtype=[("x", float, 2), ("sim_id", int), ("sim_started", bool)])
    H0["x"] = pairs
    H0["sim_id"] = range(n_samp)
    H0["sim_started"] = False

    sampling = Ensemble(parse_args=True)
    sampling.sim_specs = SimSpecs(sim_f=effis_t3d, inputs=["x"], outputs=[("Wtot_MJ", float)])
    sampling.libE_specs = LibeSpecs(sim_dirs_make=True)
    sampling.H0 = H0
    sampling.alloc_specs = AllocSpecs(alloc_f=alloc_f)
    sampling.exit_criteria = ExitCriteria(sim_max=n_samp)

    exctr = Executor()
    exctr.register_app(shutil.which("effis-submit"), app_name="effis")

    sampling.run()
    if sampling.is_manager:
        assert len(sampling.H) >= 10
        print("\nlibEnsemble with random sampling has generated enough points")
    sampling.save_output(__file__)
