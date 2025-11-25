"""
Tests libEnsemble with Optimas GridSamplingGenerator

*****currently fixing nworkers to batch_size*****

From Optimas test test_grid_sampling.py

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_optimas_grid_sample.py
   python test_optimas_grid_sample.py -n 4

When running with the above commands, the number of concurrent evaluations of
the objective function will be 4 as the generator is on the manager.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import numpy as np
from gest_api.vocs import VOCS
from optimas.generators import GridSamplingGenerator

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


def eval_func(input_params: dict):
    """Evaluation function for single-fidelity test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    return {"f": result}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    n = 2
    batch_size = 4

    libE_specs = LibeSpecs(gen_on_manager=True, nworkers=batch_size)

    # Create varying parameters.
    lower_bounds = [-3.0, 2.0]
    upper_bounds = [1.0, 5.0]
    n_steps = [7, 15]

    # Set number of evaluations.
    n_evals = np.prod(n_steps)

    vocs = VOCS(
        variables={
            "x0": [lower_bounds[0], upper_bounds[0]],
            "x1": [lower_bounds[1], upper_bounds[1]],
        },
        objectives={"f": "MAXIMIZE"},
    )

    gen = GridSamplingGenerator(vocs=vocs, n_steps=n_steps)

    gen_specs = GenSpecs(
        generator=gen,
        batch_size=batch_size,
        vocs=vocs,
    )

    sim_specs = SimSpecs(
        simulator=eval_func,
        vocs=vocs,
    )

    alloc_specs = AllocSpecs(alloc_f=alloc_f)
    exit_criteria = ExitCriteria(sim_max=n_evals)

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

        # Get generated points.
        h = H[H["sim_ended"]]
        x0_gen = h["x0"]
        x1_gen = h["x1"]

        # Get expected 1D steps along each variable.
        x0_steps = np.linspace(lower_bounds[0], upper_bounds[0], n_steps[0])
        x1_steps = np.linspace(lower_bounds[1], upper_bounds[1], n_steps[1])

        # Check that the scan along each variable is as expected.
        np.testing.assert_array_equal(np.unique(x0_gen), x0_steps)
        np.testing.assert_array_equal(np.unique(x1_gen), x1_steps)

        # Check that for every x0 step, the expected x1 steps are performed.
        for x0_step in x0_steps:
            x1_in_x0_step = x1_gen[x0_gen == x0_step]
            np.testing.assert_array_equal(x1_in_x0_step, x1_steps)

