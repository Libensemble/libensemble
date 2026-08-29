"""
Example of multi-fidelity optimization using a persistent BoTorch MFKG gen_func.

One worker runs the persistent MFKG generator; the remaining workers evaluate
the (augmented Branin) objective. The generator collects a full batch of ``q``
points before refitting its GP model, so a batch size of ``q`` keeps up to
``q`` simulation workers busy at once.

This script runs the ensemble twice to exercise both ways of returning results
to the generator:

* ``async_return=False`` (batch): the whole batch of ``q`` evaluations is
  returned to the generator in one message once all have completed.
* ``async_return=True`` with ``active_recv_gen=True``: each evaluation is
  handed back as soon as it finishes (possibly out of order); the generator
  collects the batch as results stream in.

Execute via one of the following commands:
   mpiexec -np 5 python run_botorch_mfkg_branin.py
   python run_botorch_mfkg_branin.py --nworkers 4
   python run_botorch_mfkg_branin.py --nworkers 4 --comms tcp

With ``--nworkers 4`` one worker is the generator and three concurrently
evaluate the objective.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import numpy as np

from libensemble import logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.gen_funcs.persistent_botorch_mfkg_branin import persistent_botorch_mfkg
from libensemble.libE import libE
from libensemble.sim_funcs.augmented_branin import augmented_branin
from libensemble.tools import parse_args, save_libE_output

LB = np.array([0.0, 0.0])
UB = np.array([1.0, 1.0])
N_INIT_SAMPLES = 4  # Each initial point gets a high- and a low-fidelity evaluation
Q = 4  # Batch size per MFKG iteration (keeps up to q sim workers busy)
SIM_MAX = 16  # Exit after running this many simulations


def run_mfkg(async_return, nworkers, is_manager, libE_specs):
    """Run the MFKG ensemble once and check the results.

    ``async_return`` selects batch return (False) or asynchronous return
    (True, with the generator in active-receive mode).
    """
    sim_specs = {
        "sim_f": augmented_branin,
        "in": ["x", "fidelity"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": persistent_botorch_mfkg,
        "persis_in": ["sim_id", "x", "f", "fidelity"],
        "out": [
            ("x", float, (2,)),
            ("fidelity", float),
        ],
        "user": {
            "lb": LB,
            "ub": UB,
            "n_init_samples": N_INIT_SAMPLES,
            "q": Q,
        },
    }

    alloc_specs = {
        "alloc_f": only_persistent_gens,
        "user": {
            # When async, return each evaluation to the generator as soon as it
            # completes and let the generator receive while still active. When
            # not async, the whole batch is returned in one message.
            "async_return": async_return,
            "active_recv_gen": async_return,
        },
    }

    exit_criteria = {"sim_max": SIM_MAX}

    # Fresh RNG streams for each run.
    persis_info = {}

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        save_libE_output(H, persis_info, __file__, nworkers)

        mode = "async" if async_return else "batch"

        # -- Sanity checks that the run behaved as expected --------------------
        ended = H["sim_ended"]
        n_ended = int(np.sum(ended))

        # libE finished cleanly.
        assert flag == 0, f"[{mode}] libE returned a nonzero exit flag: {flag}"

        # We completed at least sim_max evaluations, and the MFKG loop produced
        # points beyond the (2 * n_init_samples) initial sample.
        assert n_ended >= SIM_MAX, f"[{mode}] Expected >= {SIM_MAX} completed sims, got {n_ended}"
        assert n_ended > 2 * N_INIT_SAMPLES, f"[{mode}] MFKG did not generate points beyond the initial sample"

        # Every completed evaluation returned a finite objective value.
        assert np.all(np.isfinite(H["f"][ended])), f"[{mode}] Found non-finite objective value(s)"

        # The generator only ever requests the two discrete fidelities (low=0, high=1).
        fids = np.round(H["fidelity"][ended], 6)
        assert np.all(np.isin(fids, [0.0, 1.0])), f"[{mode}] Unexpected fidelity values: {np.unique(fids)}"

        # Generated points stayed within [lb, ub].
        xs = H["x"][ended]
        assert np.all(xs >= LB - 1e-9) and np.all(xs <= UB + 1e-9), f"[{mode}] Generated point(s) outside the bounds"

        # augmented_branin is negated for maximization; the best attainable value is
        # ~ -0.397887 (the negated Branin global minimum). We must never exceed it.
        best_f = float(np.max(H["f"][ended]))
        assert best_f <= -0.397887 + 1e-3, f"[{mode}] Objective exceeded the known optimum: {best_f}"
        assert best_f < 0.0, f"[{mode}] Best objective is unexpectedly non-negative: {best_f}"

        print(f"[{mode}] assertions passed: completed sims: {n_ended}, best f: {best_f:.4f}")


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    # libE logger
    logger.set_level("INFO")

    # Exercise both batch and asynchronous return of results to the generator.
    for async_return in [False, True]:
        run_mfkg(async_return, nworkers, is_manager, libE_specs)
