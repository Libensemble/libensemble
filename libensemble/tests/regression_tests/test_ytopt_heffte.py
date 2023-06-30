"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute locally via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python run_ytopt_xsbench.py
   python run_ytopt_xsbench.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 3
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import warnings

# A ytopt dependency uses an ImportWarning
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import secrets
import sys

ytopt_files_loc = "./scripts_used_by_reg_tests/ytopt-libe-speed3d/"
sys.path.append(ytopt_files_loc)

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from optimizer import Optimizer

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.ytopt_asktell import persistent_ytopt  # Gen function, communicates with ytopt optimizer

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.ytopt_obj import init_obj  # Sim function, calls Plopper
from libensemble.tools import add_unique_random_streams, parse_args

# Parse comms, default options from commandline
nworkers, is_manager, libE_specs, user_args_in = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

user_args_in = ["--learner=RF", "--max-evals=10"]
assert len(user_args_in), "learner, etc. not specified, e.g. --learner RF"
user_args = {}
for entry in user_args_in:
    if entry.startswith("--"):
        if "=" not in entry:
            key = entry.strip("--")
            value = user_args_in[user_args_in.index(entry) + 1]
        else:
            split = entry.split("=")
            key = split[0].strip("--")
            value = split[1]

    user_args[key] = value

req_settings = ["learner", "max-evals"]
assert all([opt in user_args for opt in req_settings]), "Required settings missing. Specify each setting in " + str(
    req_settings
)

# Set options so workers operate in unique directories
here = os.path.join(os.getcwd(), ytopt_files_loc)

libE_specs["use_worker_dirs"] = True
libE_specs["sim_dirs_make"] = False  # Otherwise directories separated by each sim call
libE_specs["ensemble_dir_path"] = "./ensemble_" + secrets.token_hex(nbytes=4)

# Copy or symlink needed files into unique directories
libE_specs["sim_dir_symlink_files"] = [here + f for f in ["speed3d.sh", "exe.pl", "plopper.py", "processexe.pl"]]
libE_specs["sim_dir_symlink_files"] += ["speed3d_c2c"]

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    "sim_f": init_obj,
    "in": ["p0", "p1", "p2", "p3"],
    "out": [("RUNTIME", float), ("elapsed_sec", float)],
}

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
    learner=user_args["learner"],
    liar_strategy="cl_max",
    acq_func="gp_hedge",
    set_KAPPA=1.96,
    set_SEED=2345,
    set_NI=10,
)

# Declare the gen_f that will generate points for the sim_f, and the various input/outputs
gen_specs = {
    "gen_f": persistent_ytopt,
    "out": [("p0", "<U24", (1,)), ("p1", "<U24", (1,)), ("p2", "<U24", (1,)), ("p3", "<U24", (1,))],
    "persis_in": sim_specs["in"] + ["RUNTIME"] + ["elapsed_sec"],
    "user": {
        "ytoptimizer": ytoptimizer,  # provide optimizer to generator function
        "num_sim_workers": num_sim_workers,
    },
}

alloc_specs = {
    "alloc_f": alloc_f,
    "user": {"async_return": True},
}

# Specify when to exit. More options: https://libensemble.readthedocs.io/en/main/data_structures/exit_criteria.html
exit_criteria = {"gen_max": int(user_args["max-evals"])}

# Added as a workaround to issue that's been resolved on develop
persis_info = add_unique_random_streams({}, nworkers + 1)

# Perform the libE run
H, persis_info, flag = libE(
    sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs=alloc_specs, libE_specs=libE_specs
)

# Save History array to file
if is_manager:
    print("\nlibEnsemble has completed evaluations.")
    # assert np.all(H["sim_ended"]), "Every point should have been marked as ended"
    assert len(np.unique(H["RUNTIME"])) == len(H), "Every RUNTIME should be unique"

    # save_libE_output(H, persis_info, __file__, nworkers)

    # print("\nSaving just sim_specs[['in','out']] to a CSV")
    # H = np.load(glob.glob('*.npy')[0])
    # H = H[H["sim_ended"]]
    # H = H[H["returned"]]
    # dtypes = H[gen_specs['persis_in']].dtype
    # b = np.vstack(map(list, H[gen_specs['persis_in']]))
    # print(b)
    # np.savetxt('results.csv',b, header=','.join(dtypes.names), delimiter=',',fmt=','.join(['%s']*b.shape[1]))
