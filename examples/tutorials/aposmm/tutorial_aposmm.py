import numpy as np

from tutorial_six_hump_camel import six_hump_camel

from libensemble.libE import libE
from libensemble.gen_funcs.persistent_aposmm import aposmm
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc
from libensemble.tools import parse_args, add_unique_random_streams

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

nworkers, is_manager, libE_specs, _ = parse_args()

sim_specs = {
    "sim_f": six_hump_camel,  # Simulation function
    "in": ["x"],  # Accepts 'x' values
    "out": [("f", float)],  # Returns f(x) values
}

gen_out = [
    ("x", float, 2),  # Produces 'x' values
    ("x_on_cube", float, 2),  # 'x' values scaled to unit cube
    ("sim_id", int),  # Produces IDs for sim order
    ("local_min", bool),  # Is a point a local minimum?
    ("local_pt", bool),  # Is a point from a local opt run?
]

gen_specs = {
    "gen_f": aposmm,  # APOSMM generator function
    "persis_in": ["x", "f", "x_on_cube", "sim_id", "local_min", "local_pt"],
    "out": gen_out,  # Output defined like above dict
    "user": {
        "initial_sample_size": 100,  # Random sample 100 points to start
        "localopt_method": "scipy_Nelder-Mead",
        "opt_return_codes": [0],  # Return code specific to localopt_method
        "max_active_runs": 6,  # Occur in parallel
        "lb": np.array([-2, -1]),  # Lower bound of search domain
        "ub": np.array([2, 1]),  # Upper bound of search domain
    },
}

alloc_specs = {"alloc_f": persistent_aposmm_alloc}

exit_criteria = {"sim_max": 2000}
persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
if is_manager:
    print("Minima:", H[np.where(H["local_min"])]["x"])
