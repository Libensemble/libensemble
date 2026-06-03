"""
Tests libEnsemble's capability to kill/cancel simulations that are in progress,
using the gest-api generator interface.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_surmise_killsims.py
   python test_persistent_surmise_killsims.py --nworkers 3
   python test_persistent_surmise_killsims.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.

This test is a smaller variant of test_persistent_surmise_calib.py, but which
subprocesses a compiled version of the borehole simulation. A delay is
added to simulations after the initial batch, so that the killing of running
simulations can be tested. This will only affect simulations that have already
been issued to a worker when the cancel request is registered by the manager.

See more information, see tutorial:
"Borehole Calibration with Selective Simulation Cancellation"
in the libEnsemble documentation.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4
# TESTSUITE_EXTRA: true

# Requires:
#   Install Surmise package

import os

import numpy as np
from gest_api.vocs import VOCS

from libensemble.executors.executor import Executor
from libensemble.gen_classes.surmise_calib import SurmiseCalibrator

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.borehole_kills import borehole as sim_f
from libensemble.specs import GenSpecs, SimSpecs
from libensemble.tests.regression_tests.common import build_borehole
from libensemble.tools import parse_args, save_libE_output

# from libensemble import logger
# logger.set_level("DEBUG")  # To get debug logging in ensemble.log

if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    n_init_thetas = 15  # Initial batch of thetas
    n_x = 5  # No. of x values
    nparams = 4  # No. of theta params
    ndims = 3  # No. of x coordinates.
    max_add_thetas = 20  # Max no. of thetas added for evaluation
    step_add_theta = 10  # No. of thetas to generate per step, before emulator is rebuilt
    n_explore_theta = 200  # No. of thetas to explore while selecting the next theta
    obsvar = 10 ** (-1)  # Constant for generating noise in obs

    # Batch mode until after init_sample_size (add one theta to batch for observations)
    init_sample_size = (n_init_thetas + 1) * n_x

    # Stop after max_evals evaluations
    max_evals = init_sample_size + max_add_thetas * n_x

    sim_app = os.path.join(os.getcwd(), "borehole.x")
    if not os.path.isfile(sim_app):
        build_borehole()

    exctr = Executor()  # Run serial sub-process in place
    exctr.register_app(full_path=sim_app, app_name="borehole")

    # Subprocess variant creates input and output files for each sim
    libE_specs["sim_dirs_make"] = True  # To keep all - make sim dirs
    libE_specs["kill_canceled_sims"] = True

    # Rename ensemble dir for non-interference with other regression tests
    en_suffix = str(nworkers) + "_" + libE_specs.get("comms")
    libE_specs["ensemble_dir_path"] = "ensemble_calib_kills_w" + en_suffix

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
        priorscale=0.2,
    )

    gen_out = [
        ("x", float, ndims),
        ("thetas", float, nparams),
        ("priority", int),
    ]

    sim_specs = SimSpecs(
        sim_f=sim_f,
        inputs=["x", "thetas"],
        out=[
            ("f", float),
            ("sim_killed", bool),
        ],
        user={
            "num_obs": n_x,
            "init_sample_size": init_sample_size,
        },
    )

    gen_specs = GenSpecs(
        generator=generator,
        persis_in=["f", "sim_id"],
        out=gen_out,
        initial_batch_size=init_sample_size,
        async_return=True,
        active_recv_gen=True,
    )

    exit_criteria = {"sim_max": max_evals}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, {}, libE_specs=libE_specs)

    if is_manager:
        print("Cancelled sims", H["sim_id"][H["cancel_requested"]])
        print("Kills sent by manager to running simulations", H["sim_id"][H["kill_sent"]])
        print("Killed sims", H["sim_id"][H["sim_killed"]])
        sims_done = np.count_nonzero(H["sim_ended"])
        save_libE_output(H, persis_info, __file__, nworkers)
        assert sims_done == max_evals, f"Num of completed simulations should be {max_evals}. Is {sims_done}"
