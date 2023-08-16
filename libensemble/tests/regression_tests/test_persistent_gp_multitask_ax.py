"""
Example of multi-fidelity optimization using a persistent GP gen_func (calling
Ax).

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 5 python test_persistent_gp_multitask_ax.py
   python test_persistent_gp_multitask_ax.py --nworkers 4 --comms local
   python test_persistent_gp_multitask_ax.py --nworkers 4 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as one of the three workers will be the
persistent generator.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 5
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import warnings

import numpy as np

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.message_numbers import WORKER_DONE
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, SimSpecs

# Ax uses a deprecated warn command.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    libE_output = np.zeros(1, dtype=sim_specs["out"])
    calc_status = WORKER_DONE

    # Function that depends on the resolution parameter
    libE_output["f"] = -(x0 + 10 * np.cos(x0 + 0.1 * z)) * (x1 + 5 * np.cos(x1 - 0.2 * z))

    return libE_output, persis_info, calc_status


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    mt_params = {
        "name_hifi": "expensive_model",
        "name_lofi": "cheap_model",
        "n_init_hifi": 4,
        "n_init_lofi": 4,
        "n_opt_hifi": 2,
        "n_opt_lofi": 4,
    }

    experiment = Ensemble(
        sim_specs=SimSpecs(
            sim_f=run_simulation,
            inputs=["x", "task"],
            out=[("f", float)],
        ),
        alloc_specs=AllocSpecs(
            alloc_f=only_persistent_gens,
            user={"async_return": False},
        ),
        exit_criteria=ExitCriteria(sim_max=20),
    )

    experiment.add_random_streams()
    experiment.gen_specs = GenSpecs(
        gen_f=persistent_gp_mt_ax_gen_f,
        inputs=["sim_id", "x", "f", "task"],
        persis_in=["sim_id", "x", "f", "task"],
        out=[
            # parameters to input into the simulation.
            ("x", float, (2,)),
            ("task", str, max([len(mt_params["name_hifi"]), len(mt_params["name_lofi"])])),
            ("resource_sets", int),
        ],
        user={
            "range": [1, 8],
            # Total max number of sims running concurrently.
            "gen_batch_size": experiment.nworkers - 1,
            # Lower bound for the n parameters.
            "lb": np.array([0, 0]),
            # Upper bound for the n parameters.
            "ub": np.array([15, 15]),
        },
    )
    experiment.gen_specs.user.update(**mt_params)

    experiment.run()
    experiment.save_output(__file__)
