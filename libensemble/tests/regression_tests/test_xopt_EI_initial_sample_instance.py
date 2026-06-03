"""
Tests libEnsemble with Xopt ExpectedImprovementGenerator using a
pre-constructed sampler instance for ``initial_sample_method``.

Companion to ``test_xopt_EI_initial_sample.py``, which uses the string form
(``initial_sample_method="uniform"``). This test instead passes a pre-configured
``LatinHypercubeSample`` instance — exercising the path that lets the user
supply constructor kwargs (here, ``random_seed``) and choose any sampler from
``gen_classes.sampling`` (or a custom one) without going through the string
registry in ``runners.py``.

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_xopt_EI_initial_sample_instance.py
   python test_xopt_EI_initial_sample_instance.py -n 4
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
from libensemble.gen_classes.sampling import LatinHypercubeSample
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


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

    # Pre-constructed sampler with a custom random_seed — not reachable via the
    # string form, which always instantiates with sampler defaults.
    initial_sampler = LatinHypercubeSample(vocs=vocs, random_seed=42)

    gen_specs = GenSpecs(
        generator=gen,
        initial_batch_size=batch_size,
        initial_sample_method=initial_sampler,
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

    if workflow.is_manager:
        print(f"Completed {len(H)} simulations")
        assert len(H) >= 8, f"Expected at least 8 sims, got {len(H)}"
        print("Test passed")
