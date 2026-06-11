"""
Tests libEnsemble with Xopt ExpectedImprovementGenerator using
initial_sample_method="uniform" to produce initial sample points.

EI requires pre-evaluated data before it can suggest points. This test
verifies that setting initial_sample_method="uniform" in GenSpecs causes
libEnsemble to generate uniform random samples, evaluate them through
the sim, and ingest results into the generator before optimization begins.

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_xopt_EI_initial_sample.py
   python test_xopt_EI_initial_sample.py -n 4

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true
# TESTSUITE_EXCLUDE: true

import numpy as np
from gest_api.vocs import VOCS
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.specs import AllocSpecs, GenSpecs, LibeSpecs, SimSpecs


def xtest_sim(H, persis_info, sim_specs, _):
    """y1 = x2, c1 = x1"""
    batch = len(H)
    H_o = np.zeros(batch, dtype=sim_specs["out"])
    for i in range(batch):
        H_o["y1"][i] = H["x2"][i]
        H_o["c1"][i] = H["x1"][i]
    return H_o, persis_info


if __name__ == "__main__":

    batch_size = 4

    libE_specs = LibeSpecs(gen_on_manager=True, nworkers=batch_size)
    libE_specs.reuse_output_dir = True

    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"y1": "MINIMIZE"},
        constraints={"c1": ["GREATER_THAN", 0.5]},
        constants={"constant1": 1.0},
    )

    gen = ExpectedImprovementGenerator(vocs=vocs)

    # NO pre-ingested data — libEnsemble handles initial sampling.
    gen_specs = GenSpecs(
        generator=gen,
        initial_batch_size=batch_size,
        initial_sample_method="uniform",
        batch_size=batch_size,
        vocs=vocs,
    )

    sim_specs = SimSpecs(
        sim_f=xtest_sim,
        vocs=vocs,
    )

    alloc_specs = AllocSpecs(alloc_f=alloc_f)

    workflow = Ensemble(
        libE_specs=libE_specs,
        sim_specs=sim_specs,
        alloc_specs=alloc_specs,
        gen_specs=gen_specs,
    )

    H, _, _ = workflow.run(sim_max=20)

    if workflow.is_manager:
        print(f"Completed {len(H)} simulations")
        assert len(H) >= 8, f"Expected at least 8 sims, got {len(H)}"
        print("Test passed")
