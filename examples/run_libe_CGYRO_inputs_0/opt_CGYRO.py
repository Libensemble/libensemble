#!/usr/bin/env python

import os
import sys, getopt
import subprocess

import jinja2
import numpy as np
from CGYRO_sim import run_CGYRO  # Sim func from current dir

import libensemble.gen_funcs
from libensemble import Ensemble
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.executors import MPIExecutor

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

from math import gamma, pi, sqrt


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "n:", ["nproc=", "nomp=", "numa=", "mpinuma="])

        print(opts)
        print(args)

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("--nproc"):
            nproc = arg
        elif opt in ("--nomp"):
            nomp = arg
        elif opt in ("--numa"):
            numa = arg
        elif opt in ("--mpinuma"):
            mpinuma = arg

    cgyro_input_file = "input.cgyro"

    exctr = MPIExecutor()

    # sim_app = "/global/common/software/atom/gacode-perlmutter-gpu/cgyro/src/cgyro"
    sim_app = "/global/cfs/cdirs/m4493/ebelli/gacode/cgyro/src/cgyro"
    # wrapper = "/global/cfs/cdirs/m4493/ebelli/gacode-perlmutter-gpu/platform/exec/wrap.PERLMUTTER_GPU"
    wrapper = "/global/cfs/cdirs/m4493/ebelli/gacode/platform/exec/wrap.PERLMUTTER_GPU"

    if not os.path.isfile(sim_app):
        sys.exit("cgyro not found")

    exctr.register_app(full_path=sim_app, app_name="cgyro", precedent=wrapper)

    # Parse number of workers, comms type, etc. from arguments
    ensemble = Ensemble(parse_args=True, executor=exctr)
    nworkers = ensemble.nworkers

    platform_specs = {"gpu_setting_type": "option_gpus_per_node"}

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,
        sim_dirs_make=True,
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/nl01",
        sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/reg02",
        platform_specs=platform_specs,
        reuse_output_dir=True,
        save_every_k_sims=1,
        stats_fmt = {"show_resource_sets": True, "task_datetime": True},
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_CGYRO,
        inputs=["x"],
        outputs=[("f", float), ("fvec", float, 2)],
        user={
            "input_filename": cgyro_input_file,
            "input_names": ["KAPPA"],
            "plot_heat_flux": False,
            "nproc": nproc,
            "nomp": nomp,
            "numa": numa,
            "mpinuma": mpinuma,
        },
    )
    
    n = 1
    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        inputs=[],  # No input when start persistent generator
        persis_in=["f"] + [n[0] for n in gen_out],
        outputs=[("x", float, (1,))],
        user={
            "initial_sample_size": 1,
            "sample_points": np.array([2.222]),
            "lb": np.array([0.5]),  # lower bound for input
            "ub": np.array([4.0]),  # upper bound for input
            "localopt_method": "LN_BOBYQA",
            "rk_const": 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            "xtol_abs": 1e-6,
            "ftol_abs": 1e-6,
            "dist_to_bound_multiple": 0.05,
            "max_active_runs": 1,
        },
    )

    # Starts one persistent generator. Simulated values are returned in batch.
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "async_return": False,  # False causes batch returns
        },
    )

    # Instruct libEnsemble to exit after this many simulations
    ensemble.exit_criteria = ExitCriteria(sim_max=5)

    # Seed random streams for each worker, particularly for gen_f
    ensemble.add_random_streams()

    # Run ensemble
    ensemble.run()

    if ensemble.is_manager:
        ensemble.save_output(__file__)


if __name__ == "__main__":
    main(sys.argv[1:])
