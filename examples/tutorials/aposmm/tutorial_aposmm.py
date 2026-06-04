import multiprocessing

import numpy as np
from gest_api.vocs import VOCS

import libensemble.gen_funcs
from libensemble import Ensemble
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

from libensemble.gen_classes import APOSMM  # noqa

multiprocessing.set_start_method("fork", force=True)


def six_hump_camel(x):
    """Six-hump camel function, gest-api style (dict in, dict out)."""
    x1 = x["x0"]
    x2 = x["x1"]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return {"f": term1 + term2 + term3}


vocs = VOCS(
    variables={
        "x0": [-2, 2],
        "x1": [-1, 1],
        "x0_cube": [0, 1],
        "x1_cube": [0, 1],
    },
    objectives={"f": "MINIMIZE"},
)

aposmm = APOSMM(
    vocs,
    max_active_runs=6,
    initial_sample_size=100,
    variables_mapping={
        "x": ["x0", "x1"],
        "x_on_cube": ["x0_cube", "x1_cube"],
        "f": ["f"],
    },
    localopt_method="scipy_Nelder-Mead",
    opt_return_codes=[0],
)

workflow = Ensemble(parse_args=True)
workflow.gen_specs = GenSpecs(generator=aposmm, vocs=vocs, batch_size=6, initial_batch_size=100)
workflow.sim_specs = SimSpecs(simulator=six_hump_camel, vocs=vocs)
workflow.exit_criteria = ExitCriteria(sim_max=2000)

H, _, _ = workflow.run()

if workflow.is_manager:
    print("Minima:", H[np.where(H["local_min"])]["x"])
