"""
Tests libEnsemble with Xopt NelderMeadGenerator using Rosenbrock function

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_xopt_nelder_mead.py
   python test_xopt_nelder_mead.py -n 4

When running with the above commands, the number of concurrent evaluations of
the objective function will be 4 as the generator is on the manager.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2
# TESTSUITE_EXTRA: true

import numpy as np
from gest_api.vocs import VOCS
from xopt.generators.sequential.neldermead import NelderMeadGenerator

from libensemble import Ensemble
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


def rosenbrock_callable(input_dict: dict) -> dict:
    """2D Rosenbrock function for gest-api style simulator"""
    x1 = input_dict["x1"]
    x2 = input_dict["x2"]
    y1 = 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2
    return {"y1": y1}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    batch_size = 1

    libE_specs = LibeSpecs(gen_on_manager=True, nworkers=batch_size)

    vocs = VOCS(
        variables={"x1": [-2.0, 2.0], "x2": [-2.0, 2.0]},
        objectives={"y1": "MINIMIZE"},
    )

    gen = NelderMeadGenerator(vocs=vocs)

    # Create initial points with evaluated rosenbrock values
    initial_points = [
        {"x1": -1.2, "x2": 1.0, "y1": rosenbrock_callable({"x1": -1.2, "x2": 1.0})["y1"]},
        {"x1": -1.0, "x2": 1.0, "y1": rosenbrock_callable({"x1": -1.0, "x2": 1.0})["y1"]},
        {"x1": -0.8, "x2": 0.8, "y1": rosenbrock_callable({"x1": -0.8, "x2": 0.8})["y1"]},
    ]
    gen.ingest(initial_points)

    gen_specs = GenSpecs(
        generator=gen,
        batch_size=batch_size,
        vocs=vocs,
    )

    sim_specs = SimSpecs(
        simulator=rosenbrock_callable,
        vocs=vocs,
    )

    exit_criteria = ExitCriteria(sim_max=30)

    workflow = Ensemble(
        libE_specs=libE_specs,
        sim_specs=sim_specs,
        gen_specs=gen_specs,
        exit_criteria=exit_criteria,
    )

    H, _, _ = workflow.run()

    # Perform the run
    if workflow.is_manager:
        print(f"Completed {len(H)} simulations")
        initial_value = H["y1"][0]
        best_value = H["y1"][np.argmin(H["y1"])]
        assert best_value <= initial_value
