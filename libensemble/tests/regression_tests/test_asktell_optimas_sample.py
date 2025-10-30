"""
Runs libEnsemble with APOSMM with the NLopt local optimizer.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_nlopt.py
   python test_persistent_aposmm_nlopt.py --nworkers 3 --comms local
   python test_persistent_aposmm_nlopt.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import sys
from time import time

from gest_api.vocs import VOCS
from optimas.generators import GridSamplingGenerator

# Import libEnsemble items for this test
from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, SimSpecs

if __name__ == "__main__":

    workflow = Ensemble(parse_args=True)

    if workflow.is_manager:
        start_time = time()

    if workflow.nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2

    workflow.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], outputs=[("f", float)])
    workflow.alloc_specs = AllocSpecs(alloc_f=alloc_f)
    workflow.exit_criteria = ExitCriteria(sim_max=35)

    vocs = VOCS(
        variables={"x0": [-3, 3], "x1": [-2, 2]},
        objectives={"f": "MINIMIZE"},
    )

    # Generator from Optimas
    gen = GridSamplingGenerator(vocs, n_steps=[5, 7])

    workflow.gen_specs = GenSpecs(
        persis_in=["x", "f"],
        outputs=[("x", float, 2)],
        generator=gen,
        batch_size=5,
        initial_batch_size=10,
    )

    workflow.libE_specs.gen_on_manager = True
    workflow.add_random_streams()

    H, _, _ = workflow.run()

    if workflow.is_manager:
        print("[Manager]: Time taken =", time() - start_time, flush=True)
        print("[Manager]:", H[["x", "f"]])
