"""
Runs libEnsemble with Latin hypercube sampling and check no warning.

Execute using MPI (e.g. 3 workers):
   mpiexec -np 4 python test_mpi_warning.py

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 4

import os
import time

import numpy as np

from libensemble import Ensemble, logger
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f

# Import libEnsemble items for this test
from libensemble.sim_funcs.simple_sim import norm_eval as sim_f
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    log_file = "ensemble_check_warning.log"
    logger.set_level("MANAGER_WARNING")
    logger.set_filename(log_file)

    sampling = Ensemble()
    sampling.libE_specs.save_every_k_sims = 100
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

    sampling.exit_criteria = ExitCriteria(sim_max=100)
    sampling.add_random_streams()

    if sampling.is_manager:
        if os.path.exists(log_file):
            os.remove(log_file)

    sampling.run()
    if sampling.is_manager:
        print("len:", len(sampling.H))
        time.sleep(0.2)
        assert os.path.exists(log_file)
        assert os.stat(log_file).st_size == 0, "Unexpected warning"
