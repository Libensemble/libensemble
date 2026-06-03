"""
Runs libEnsemble with Surmise calibration test using the gest-api generator.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_surmise_calib.py
   python test_persistent_surmise_calib.py --nworkers 3
   python test_persistent_surmise_calib.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.

This test uses the Surmise package to perform a Borehole Calibration with
selective simulation cancellation. Initial observations are modeled using
a theta at the center of a unit hypercube. The initial function values for
these are run first. As the model is updated, the generator selects previously
issued evaluations to cancel.

See more information, see tutorial:
"Borehole Calibration with Selective Simulation Cancellation"
in the libEnsemble documentation.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

# Requires:
#   Install Surmise package

import sys

import numpy as np
from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes.surmise_calib import SurmiseCalibrator
from libensemble.sim_funcs.surmise_test_function import borehole as sim_f
from libensemble.sim_funcs.surmise_test_function import tstd2theta
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs
from libensemble.tools import parse_args


def run_surmise_calib():
    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n_init_thetas = 15  # Initial batch of thetas
    n_x = 25  # No. of x values
    nparams = 4  # No. of theta params
    ndims = 3  # No. of x coordinates.
    max_add_thetas = 50  # Max no. of thetas added for evaluation
    step_add_theta = 10  # No. of thetas to generate per step, before emulator is rebuilt
    n_explore_theta = 200  # No. of thetas to explore while selecting the next theta
    obsvar = 10 ** (-1)  # Constant for generating noise in obs

    # Batch mode until after init_sample_size (add one theta to batch for observations)
    init_sample_size = (n_init_thetas + 1) * n_x

    # Stop after max_evals evaluations
    max_evals = init_sample_size + max_add_thetas * n_x

    # Define the problem via VOCS
    vocs = VOCS(
        variables={
            "x0": [0, 1.0],
            "x1": [0, 1.0],
            "x2": [0, 1.0],
            "theta0": [0, 1.0],
            "theta1": [0, 1.0],
            "theta2": [0, 1.0],
            "theta3": [0, 1.0],
        },
        objectives={"f": "EXPLORE"},
    )

    # Initialize the standardized generator
    generator = SurmiseCalibrator(
        vocs,
        n_init_thetas=n_init_thetas,
        num_x_vals=n_x,
        step_add_theta=step_add_theta,
        n_explore_theta=n_explore_theta,
        obsvar=obsvar,
        priorloc=1,
        priorscale=0.5,
    )

    gen_out = [
        ("x", float, ndims),
        ("thetas", float, nparams),
        ("priority", int),
    ]

    test = Ensemble(
        parse_args=True,
        sim_specs=SimSpecs(
            sim_f=sim_f,
            inputs=["x", "thetas"],
            out=[("f", float)],
            user={"num_obs": n_x},
        ),
        gen_specs=GenSpecs(
            generator=generator,
            persis_in=["f", "sim_id"],
            out=gen_out,
            initial_batch_size=init_sample_size,
            async_return=True,
            active_recv_gen=True,
        ),
        exit_criteria=ExitCriteria(sim_max=max_evals),
    )

    # Perform the run
    H, _, _ = test.run()

    if test.is_manager:
        print("Cancelled sims", H["sim_id"][H["cancel_requested"]])
        sims_done = np.count_nonzero(H["sim_ended"])
        test.save_output(__file__)
        assert sims_done == max_evals, f"No. of completed simulations should be {max_evals}. Is {sims_done}"

        # The following line is only to cover parts of tstd2theta
        tstd2theta(H[0]["thetas"].squeeze(), hard=False)


if __name__ == "__main__":
    run_surmise_calib()
