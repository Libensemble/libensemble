import numpy as np
from libensemble.alloc_funcs import defaults as alloc_defaults
from libensemble.history import History


# -------------------------------------------------------------------------------------------------
# Set up sim_specs, gen_specs, exit_criteria


def make_criteria_and_specs_0(simx=10, n=1):
    sim_specs = {
        "sim_f": np.linalg.norm,
        "in": ["x_on_cube"],
        "persis_in": [],
        "funcx_endpoint": "",
        "out": [("f", float), ("fvec", float, 3)],
    }
    gen_specs = {
        "gen_f": np.random.uniform,
        "in": [],
        "persis_in": [],
        "funcx_endpoint": "",
        "out": [("priority", float), ("local_pt", bool), ("local_min", bool), ("num_active_runs", int)],
        "user": {"ub": np.ones(n), "lb": np.zeros(n), "nu": 0},
    }
    if n == 1:
        gen_specs["out"] += [("x_on_cube", float)]
    else:
        gen_specs["out"] += [("x_on_cube", float, n)]
    exit_criteria = {"sim_max": simx}

    return sim_specs, gen_specs, exit_criteria


def make_criteria_and_specs_1(simx=10):
    sim_specs = {"sim_f": np.linalg.norm, "in": ["x"], "out": [("g", float)]}
    gen_specs = {"gen_f": np.random.uniform, "in": [], "out": [("x", float), ("priority", float)], "user": {}}
    exit_criteria = {"sim_max": simx, "stop_val": ("g", -1), "wallclock_max": 0.5}

    return sim_specs, gen_specs, exit_criteria


def make_criteria_and_specs_1A(simx=10):
    sim_specs = {"sim_f": np.linalg.norm, "in": ["x"], "out": [("g", float)]}
    gen_specs = {
        "gen_f": np.random.uniform,
        "in": [],
        "out": [("x", float), ("priority", float), ("sim_id", int)],
        "user": {},
    }
    exit_criteria = {"sim_max": simx, "stop_val": ("g", -1), "wallclock_max": 0.5}

    return sim_specs, gen_specs, exit_criteria


# -------------------------------------------------------------------------------------------------
# Set up history array
def hist_setup1(sim_max=10, n=1, H0_in=[]):
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_0(simx=sim_max, n=n)
    alloc_specs = alloc_defaults.alloc_specs
    H0 = H0_in
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    return hist, sim_specs, gen_specs, exit_criteria, alloc_specs


def hist_setup2(sim_max=10, H0_in=[]):
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1(simx=sim_max)
    alloc_specs = alloc_defaults.alloc_specs
    H0 = H0_in
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    return hist, sim_specs, gen_specs, exit_criteria, alloc_specs


def hist_setup2A_genout_sim_ids(sim_max=10):
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1A(simx=sim_max)
    alloc_specs = alloc_defaults.alloc_specs
    H0 = []
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    return hist, sim_specs, gen_specs, exit_criteria, alloc_specs
