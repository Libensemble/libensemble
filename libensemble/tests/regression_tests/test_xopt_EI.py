"""
Tests libEnsemble with Xopt ExpectedImprovementGenerator

*****currently fixing nworkers to batch_size*****

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_xopt_EI.py
   python test_xopt_EI.py -n 4

When running with the above commands, the number of concurrent evaluations of
the objective function will be 4 as the generator is on the manager.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import numpy as np
from gest_api.vocs import VOCS
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


# SH TODO - should check constant1 is present
# Adapted from Xopt/xopt/resources/testing.py
def xtest_sim(H, persis_info, sim_specs, _):
    """
    Simple sim function that takes x1, x2, constant1 from H and returns y1, c1.
    Logic: y1 = x2, c1 = x1
    """
    batch = len(H)
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i in range(batch):
        x1 = H["x1"][i]
        x2 = H["x2"][i]
        # constant1 is available but not used in the calculation

        H_o["y1"][i] = x2
        H_o["c1"][i] = x1

    return H_o, persis_info


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    n = 2
    batch_size = 4

    libE_specs = LibeSpecs(gen_on_manager=True, nworkers=batch_size)

    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"y1": "MINIMIZE"},
        constraints={"c1": ["GREATER_THAN", 0.5]},
        constants={"constant1": 1.0},
    )

    gen = ExpectedImprovementGenerator(vocs=vocs)

    # Create 4 initial points and ingest them
    initial_points = [
        {"x1": 0.2, "x2": 2.0, "constant1": 1.0, "y1": 2.0, "c1": 0.2},
        {"x1": 0.5, "x2": 5.0, "constant1": 1.0, "y1": 5.0, "c1": 0.5},
        {"x1": 0.7, "x2": 7.0, "constant1": 1.0, "y1": 7.0, "c1": 0.7},
        {"x1": 0.9, "x2": 9.0, "constant1": 1.0, "y1": 9.0, "c1": 0.9},
    ]
    gen.ingest(initial_points)

    gen_specs = GenSpecs(
        generator=gen,
        batch_size=batch_size,
        vocs=vocs,
    )

    sim_specs = SimSpecs(
        sim_f=xtest_sim,
        vocs=vocs,
    )

    alloc_specs = AllocSpecs(alloc_f=alloc_f)
    exit_criteria = ExitCriteria(sim_max=20)

    workflow = Ensemble(
        libE_specs=libE_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        gen_specs=gen_specs,
        exit_criteria=exit_criteria,
    )

    H, _, _ = workflow.run()

    # Perform the run
    if workflow.is_manager:
        print(f"Completed {len(H)} simulations")
        assert np.array_equal(H["y1"], H["x2"])
        assert np.array_equal(H["c1"], H["x1"])
