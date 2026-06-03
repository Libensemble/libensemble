"""
Tests libEnsemble with BoTorchMFKG generator (gest-api style) and the
augmented Branin multi-fidelity simulator.

The generator runs on the manager thread (default). All allocated workers
are available for simulation tasks.

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_botorch_mfkg_branin.py
   python test_botorch_mfkg_branin.py -n 4

When running with the above commands, the number of concurrent evaluations
of the objective function will be 4.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes.botorch_mfkg import BoTorchMFKG
from libensemble.sim_funcs.augmented_branin import augmented_branin_callable
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

# Main block is necessary only when using local comms with spawn start method
# (default on macOS and Windows).
if __name__ == "__main__":
    # q candidates per MFKG iteration; batch_size must match so libEnsemble
    # always requests exactly the number the generator will produce.
    q = 2
    n_workers = 4

    libE_specs = LibeSpecs(nworkers=n_workers)

    # Variables: two design dimensions + fidelity (all in [0, 1]).
    # Objective: maximise the (negated) augmented Branin value.
    vocs = VOCS(
        variables={"x0": [0.0, 1.0], "x1": [0.0, 1.0], "fidelity": [0.0, 1.0]},
        objectives={"f": "MAXIMIZE"},
    )

    gen = BoTorchMFKG(
        vocs=vocs,
        n_init_samples=4,  # produces 2 * 4 = 8 initial simulations
        q=q,
        num_fantasies=128,
        num_restarts=1,  # reduced for faster testing
        raw_samples=10,  # reduced for faster testing
        seed=42,
    )

    gen_specs = GenSpecs(
        generator=gen,
        # initial_batch_size matches the 2 * n_init_samples points suggest()
        # returns on the first call.
        initial_batch_size=2 * gen.n_init_samples,
        batch_size=q,
        vocs=vocs,
    )

    sim_specs = SimSpecs(
        simulator=augmented_branin_callable,
        vocs=vocs,
    )

    # Exit after running sim_max simulations (8 initial + at least 2 iterations)
    exit_criteria = ExitCriteria(sim_max=12)

    workflow = Ensemble(
        libE_specs=libE_specs,
        sim_specs=sim_specs,
        gen_specs=gen_specs,
        exit_criteria=exit_criteria,
    )

    H, _, flag = workflow.run()

    if workflow.is_manager:
        print(f"Completed {len(H)} simulations")
        assert len(H) >= exit_criteria.sim_max, f"Expected at least {exit_criteria.sim_max} simulations, got {len(H)}"
        print("Test passed")
