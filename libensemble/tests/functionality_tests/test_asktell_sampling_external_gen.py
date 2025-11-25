"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

using external gest_api compatible generators.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_asktell_sampling_external_gen.py
   python test_asktell_sampling_external_gen.py --nworkers 3 --comms local
   python test_asktell_sampling_external_gen.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import numpy as np
from gest_api.vocs import VOCS

# from gest_api.vocs import ContinuousVariable

# Import libEnsemble items for this test
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f

# from libensemble.gen_classes.external.sampling import UniformSampleArray
from libensemble.gen_classes.external.sampling import UniformSample
from libensemble import Ensemble
from libensemble.specs import GenSpecs, SimSpecs, AllocSpecs, ExitCriteria, LibeSpecs


def sim_f_array(In):
    Out = np.zeros(1, dtype=[("f", float)])
    Out["f"] = np.linalg.norm(In)
    return Out


def sim_f_scalar(In):
    Out = np.zeros(1, dtype=[("f", float)])
    Out["f"] = np.linalg.norm(In["x0"], In["x1"])
    return Out


if __name__ == "__main__":

    libE_specs = LibeSpecs(gen_on_manager=True)

    for test in range(1):  # 2

        objectives = {"f": "EXPLORE"}

        if test == 0:
            sim_f = sim_f_scalar
            variables = {"x0": [-3, 3], "x1": [-2, 2]}
            vocs = VOCS(variables=variables, objectives=objectives)
            generator = UniformSample(vocs)

        # Requires gest-api variables array bounds update
        # elif test == 1:
        #     sim_f = sim_f_array
        #     variables = {"x": ContinuousVariable(dtype=(float, (2,)),domain=[[-3, 3], [-2, 2]])}
        #     vocs = VOCS(variables=variables, objectives=objectives)
        #     generator = UniformSampleArray(vocs)

        sim_specs = SimSpecs(
            sim_f=sim_f,
            vocs=vocs,
        )

        gen_specs = GenSpecs(
            generator=generator,
            initial_batch_size=20,
            batch_size=10,
            vocs=vocs,
        )

        alloc_specs = AllocSpecs(alloc_f=alloc_f)
        exit_criteria = ExitCriteria(gen_max=201)

        ensemble = Ensemble(
            parse_args=True,
            sim_specs=sim_specs,
            gen_specs=gen_specs,
            exit_criteria=exit_criteria,
            alloc_specs=alloc_specs,
            libE_specs=libE_specs,
        )

        ensemble.add_random_streams()
        ensemble.run()

        if ensemble.is_manager:
            print(ensemble.H[["sim_id", "x0", "x1", "f"]][:10])
            # print(ensemble.H[["sim_id", "x", "f"]][:10])  # For array variables
            assert len(ensemble.H) >= 201, f"H has length {len(ensemble.H)}"
