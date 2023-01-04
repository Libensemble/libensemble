"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the rosenbrock function as a (synthetic) simulator function.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.heffte import call_and_process_heffte as sim_f
from libensemble.gen_funcs.ytopt_gen_xsbench import persistent_ytopt as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ytopt.search.optimizer import Optimizer

nworkers, is_manager, libE_specs, user_args_in = parse_args()
user_args_in = ["learner=RF", "max-evals=10"]
if len(user_args_in):
    user_args = {}
    for entry in user_args_in:
        sp = entry.split("=")
        assert len(sp) == 2, "Incorrect arg format"
        field = sp[0]
        value = sp[1]
        user_args[field] = value

num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    "sim_f": sim_f,
    "in": ["p0", "p1", "p2", "p3"],
    "out": [("RUN_TIME", float)],
}

# Initialize the ytopt ask/tell interface (to be used by the gen_f)
cs = CS.ConfigurationSpace(seed=1234)
p0 = CSH.CategoricalHyperparameter(name="p0", choices=["-no-reorder", "-reorder"], default_value="-no-reorder")
p1 = CSH.CategoricalHyperparameter(name="p1", choices=["-a2a", "-a2av", "-p2p", "-p2p_pl"], default_value="-a2a")
p2 = CSH.CategoricalHyperparameter(name="p2", choices=["-ingrid 4 1 1", "-ingrid 2 2 1"], default_value="-ingrid 4 1 1")
p3 = CSH.CategoricalHyperparameter(
    name="p3", choices=["-outgrid 4 1 1", "-outgrid 2 2 1"], default_value="-outgrid 4 1 1"
)
cs.add_hyperparameters([p0, p1, p2, p3])

ytoptimizer = Optimizer(
    num_workers=num_sim_workers,
    space=cs,
    learner="RF",
    liar_strategy="cl_max",
    acq_func="gp_hedge",
    set_KAPPA=1.96,
    set_SEED=12345,
    set_NI=10,
)

# Declare the gen_f that will generator points for the sim_f, and the various input/outputs
gen_specs = {
    "gen_f": gen_f,
    "out": [("p0", str, 20), ("p1", str, 20), ("p2", str, 20), ("p3", str, 20)],
    "persis_in": sim_specs["in"] + ["RUN_TIME", "sim_ended_time", "sim_started_time"],
    "user": {
        "ytoptimizer": ytoptimizer,
        "num_sim_workers": num_sim_workers,
    },
}

alloc_specs = {
    "alloc_f": alloc_f,
    "user": {"async_return": True},
}

exit_criteria = {"sim_max": 10}

# Perform the libE run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs, libE_specs=libE_specs)

if is_manager:
    assert np.sum(H["sim_ended"]) == exit_criteria["sim_max"]
    print("\nlibEnsemble has perform the correct number of evaluations")
    save_libE_output(H, persis_info, __file__, nworkers)
