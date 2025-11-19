"""
Tests libEnsemble with Xopt ExpectedImprovementGenerator

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_xopt_EI.py
   python test_xopt_EI.py --nworkers 3 --comms local

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3 as the generator is on the manager.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import sys
import warnings

import numpy as np
from gest_api.vocs import VOCS
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator

from libensemble import Ensemble
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

warnings.filterwarnings("ignore", message="Default hyperparameter_bounds")


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    n = 2
    # batch_size = 15
    # num_batches = 10

    libE_specs = LibeSpecs(gen_on_manager=True)
    
    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"y1": "MINIMIZE"},
        constraints={"c1": ["GREATER_THAN", 0.5]},
        constants={"constant1": 1.0},     
    )

    gen = ExpectedImprovementGenerator(vocs=vocs)

    # SH TODO - We must enable this to be set by VOCS
    gen_specs = GenSpecs(
        persis_in=["x", "f", "sim_id"],
        out=[("x", float, (n,))],
        # batch_size=batch_size,
        generator=gen,
        user={
            "lb": np.array([0,0]),
            "ub": np.array([0,10.0]),
        },
    )

    sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], outputs=[("f", float)])
    alloc_specs = AllocSpecs(alloc_f=alloc_f)
    exit_criteria = ExitCriteria(sim_max=20)

    workflow = Ensemble(
        parse_args=True,
        libE_specs=libE_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        gen_specs=gen_specs,
        exit_criteria=exit_criteria,
    )

    H, _, _ = workflow.run()

    # Perform the run
    if workflow.is_manager:
        assert len(np.unique(H["gen_ended_time"])) == num_batches
