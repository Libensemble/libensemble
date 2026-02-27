"""
Tests libEnsemble with Optimas Single-Fidelity Ax Generator

*****currently fixing nworkers to batch_size*****

Execute via one of the following commands (e.g. 4 workers):
   mpiexec -np 5 python test_optimas_ax_sf.py
   python test_optimas_ax_sf.py -n 4

When running with the above commands, the number of concurrent evaluations of
the objective function will be 4 as the generator is on the manager.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import numpy as np
from gest_api.vocs import VOCS
from optimas.generators import AxSingleFidelityGenerator

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs


def eval_func_sf(input_params):
    """Evaluation function for single-fidelity test."""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    return {"f": result}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    n = 2
    batch_size = 2

    libE_specs = LibeSpecs(gen_on_manager=True, nworkers=batch_size)

    vocs = VOCS(
        variables={
            "x0": [-50.0, 5.0],
            "x1": [-5.0, 15.0],
        },
        objectives={"f": "MAXIMIZE"},
    )

    gen = AxSingleFidelityGenerator(vocs=vocs)

    gen_specs = GenSpecs(
        generator=gen,
        batch_size=batch_size,
        vocs=vocs,
    )

    sim_specs = SimSpecs(
        simulator=eval_func_sf,
        vocs=vocs,
    )

    alloc_specs = AllocSpecs(alloc_f=alloc_f)
    exit_criteria = ExitCriteria(sim_max=10)

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
        workflow.save_output(__file__)
        print(f"Completed {len(H)} simulations")
