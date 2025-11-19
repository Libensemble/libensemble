"""
Tests libEnsemble with Xopt ExpectedImprovementGenerator

*****currently fixing nworkers to batch_size*****

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
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
import pdb_si

warnings.filterwarnings("ignore", message="Default hyperparameter_bounds")


# SH TODO - should check constant1 is present 
# From Xopt/xopt/resources/testing.py
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

    # SH TODO - We must enable this to be set by VOCS
    gen_specs = GenSpecs(
        # initial_batch_size=4,
        generator=gen,
        batch_size=batch_size,
        vocs=vocs,
    )

    #SH TEMP PRINTS TO CHECK VOCS WORKING
    print(f'gen_specs.persis_in: {gen_specs.persis_in}')
    print(f'gen_specs.outputs: {gen_specs.outputs}')

    # SH TODO - We must enable this to be set by VOCS
    sim_specs = SimSpecs(
        sim_f=xtest_sim,
        vocs=vocs,
    )
    
    #SH TEMP PRINTS TO CHECK VOCS WORKING
    print(f'sim_specs.inputs: {sim_specs.inputs}')
    print(f'sim_specs.outputs: {sim_specs.outputs}')

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
