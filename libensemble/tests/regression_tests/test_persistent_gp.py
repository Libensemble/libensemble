"""
Example of multi-fidelity optimization using a persistent GP gen_func (calling
dragonfly) and an algebraic sim_f (that doesn't change with the amount of
resources give). Tests both with and without using an initial H0.

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 5 python test_persistent_gp.py
   python test_persistent_gp.py --nworkers 4 --comms local
   python test_persistent_gp.py --nworkers 4 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 5
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import os
import time
import warnings

import numpy as np

from libensemble import logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.gen_funcs.persistent_gp import persistent_gp_gen_f, persistent_gp_mf_disc_gen_f, persistent_gp_mf_gen_f
from libensemble.libE import libE
from libensemble.message_numbers import WORKER_DONE
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Dragonfly uses a deprecated np.asscalar command.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_simulation(H, persis_info, sim_specs, libE_info):
    # Extract input parameters
    values = list(H["x"][0])
    x0 = values[0]
    x1 = values[1]
    # Extract fidelity parameter
    z = H["z"][0]

    libE_output = np.zeros(1, dtype=sim_specs["out"])
    calc_status = WORKER_DONE

    # Function that depends on the resolution parameter
    libE_output["f"] = -(x0 + 10 * np.cos(x0 + 0.1 * z)) * (x1 + 5 * np.cos(x1 - 0.2 * z))

    return libE_output, persis_info, calc_status


def cost(z):
    return z[0]


def cost1(z):
    return z[0][0] ** 3


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    assert nworkers == 4, "This test requires exactly 4 workers"

    sim_specs = {
        "sim_f": run_simulation,
        "in": ["x", "z"],
        "out": [("f", float)],
    }

    gen_specs = {
        # Generator function. Will randomly generate new sim inputs 'x'.
        "gen_f": persistent_gp_gen_f,
        # Generator input. This is a RNG, no need for inputs.
        "persis_in": ["sim_id", "x", "f", "z"],
        "out": [
            # parameters to input into the simulation.
            ("x", float, (2,)),
            ("z", float),
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

    alloc_specs = {
        "alloc_f": only_persistent_gens,
        "user": {"async_return": True},
    }

    # libE logger
    logger.set_level("INFO")

    persis_info = add_unique_random_streams({}, nworkers + 1)

    outfile = None
    # Run LibEnsemble, and store results in history array H
    for use_H0 in [False, True]:
        if use_H0:
            if libE_specs["comms"] == "mpi":  # Want to make sure manager has saved output
                libE_specs["mpi_comm"].Barrier()
            while not os.path.isfile("gp_out.npy"):
                time.sleep(0.01)
            # other MPI processes don't have shared memory don't know outfile has been updated?
            H0 = np.load("gp_out.npy")
            H0 = H0[:6]
            gen_specs["in"] = list(H0.dtype.names)
            exit_criteria = {"sim_max": 5}  # Do 5 more evaluations
        else:
            H0 = None
            # Exit criteria
            exit_criteria = {"sim_max": 6}  # Exit after running sim_max simulations

        for run in range(3):
            # Create a different random number stream for each worker and the manager
            persis_info = add_unique_random_streams({}, nworkers + 1)

            if run == 0:
                gen_specs["gen_f"] = persistent_gp_gen_f
                gen_specs["user"]["cost_func"] = cost
            if run == 1:
                gen_specs["gen_f"] = persistent_gp_mf_gen_f
                gen_specs["user"]["cost_func"] = cost
            elif run == 2:
                gen_specs["gen_f"] = persistent_gp_mf_disc_gen_f
                gen_specs["user"]["cost_func"] = cost1

            H, persis_info, flag = libE(
                sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0=H0
            )

            if is_manager:
                if use_H0 is False:
                    if run == 0:
                        assert not len(np.unique(H["resource_sets"])) > 1, "Resource sets should be the same"
                        outfile = save_libE_output(H, persis_info, __file__, nworkers)
                        os.rename(outfile, "gp_out.npy")
                    else:
                        print(H["resource_sets"])
                        assert len(np.unique(H["resource_sets"])) > 1, "Resource sets should be variable."
