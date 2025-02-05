"""
Example of multi-fidelity optimization using a persistent GP gen_func (calling
Ax).

Test is set to use the gen_on_manager option (persistent generator runs on
a thread). Therefore nworkers is the number of simulation workers.

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 4 python test_persistent_gp_multitask_ax.py
   python test_persistent_gp_multitask_ax.py --nworkers 3 --comms local
   python test_persistent_gp_multitask_ax.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import warnings

import numpy as np

from libensemble import logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.libE import libE
from libensemble.message_numbers import WORKER_DONE
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Ax uses a deprecated warn command.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from libensemble.gen_funcs.persistent_ax_multitask import persistent_gp_mt_ax_gen_f


def run_simulation(H, persis_info, sim_specs, libE_info):
    # Extract input parameters
    values = list(H["x"][0])
    x0 = values[0]
    x1 = values[1]
    # Extract fidelity parameter
    task = H["task"][0]
    if task == "expensive_model":
        z = 8
    elif task == "cheap_model":
        z = 1
    print('in sim', task)

    libE_output = np.zeros(1, dtype=sim_specs["out"])
    calc_status = WORKER_DONE

    # Function that depends on the resolution parameter
    libE_output["f"] = -(x0 + 10 * np.cos(x0 + 0.1 * z)) * (x1 + 5 * np.cos(x1 - 0.2 * z))

    return libE_output, persis_info, calc_status


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["gen_on_manager"] = True

    mt_params = {
        "name_hifi": "expensive_model",
        "name_lofi": "cheap_model",
        "n_init_hifi": 4,
        "n_init_lofi": 4,
        "n_opt_hifi": 2,
        "n_opt_lofi": 4,
    }

    sim_specs = {
        "sim_f": run_simulation,
        "in": ["x", "task"],
        "out": [("f", float)],
    }

    gen_specs = {
        # Generator function. Will randomly generate new sim inputs 'x'.
        "gen_f": persistent_gp_mt_ax_gen_f,
        # Generator input. This is a RNG, no need for inputs.
        "in": ["sim_id", "x", "f", "task"],
        "persis_in": ["sim_id", "x", "f", "task"],
        "out": [
            # parameters to input into the simulation.
            ("x", float, (2,)),
            ("task", str, max([len(mt_params["name_hifi"]), len(mt_params["name_lofi"])])),
            ("resource_sets", int),
        ],
        "user": {
            "range": [1, 8],
            # Total max number of sims running concurrently.
            "gen_batch_size": nworkers - 1,
            # Lower bound for the n parameters.
            "lb": np.array([0, 0]),
            # Upper bound for the n parameters.
            "ub": np.array([15, 15]),
        },
    }
    gen_specs["user"] = {**gen_specs["user"], **mt_params}

    alloc_specs = {
        "alloc_f": only_persistent_gens,
        "user": {"async_return": False},
    }

    # libE logger
    logger.set_level("INFO")

    # Exit criteria
    exit_criteria = {"sim_max": 20}  # Exit after running sim_max simulations

    # Create a different random number stream for each worker and the manager
    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Run LibEnsemble, and store results in history array H
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    # Save results to numpy file
    if is_manager:
        save_libE_output(H, persis_info, __file__, nworkers)
