#!/usr/bin/env python

import os
import sys, getopt
import subprocess

import jinja2
import numpy as np
from CGYRO_sim import run_CGYRO_over_KY  # Sim func from current dir

import libensemble.gen_funcs
from libensemble import Ensemble
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.executors import MPIExecutor

libensemble.gen_funcs.rc.aposmm_optimizers = "ibcdfo_manifold_sampling"

from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

# from ibcdfo.manifold_sampling.h_examples import pw_maximum as hfun
from ibcdfo.manifold_sampling.h_examples import max_gamma_over_KY as hfun


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
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/reg02",
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/kappa_correction",
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/kappa_correction_with_KY",
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/kappa_correction_with_KY_Belli",
        sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/kappa_correction_with_KY_Belli_two",
        platform_specs=platform_specs,
        # reuse_output_dir=True,
        save_every_k_sims=1,
        stats_fmt = {"show_resource_sets": True, "task_datetime": True},
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_CGYRO_over_KY,
        inputs=["x"],
        outputs=[("f", float), ("fvec", float, 11), ("convstatement", "U100")],
        user={
            "input_filename": cgyro_input_file,
            "input_names": ["KAPPA","DELTA","ZETA","KY"],
            "plot_heat_flux": False,
            "nproc": nproc,
            "nomp": nomp,
            "numa": numa,
            "mpinuma": mpinuma,
        },
    )
    
    n = 3
    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
        ("started_run", bool),
    ]

    ensemble.gen_specs = GenSpecs(
        gen_f= gen_f,
        persis_in= ["f", "fvec"] + [n[0] for n in gen_out],
        out= gen_out,
        user= {
            "initial_sample_size": 1,
            "stop_after_k_runs": 1,
            "max_active_runs": 1,
            "sample_points": np.atleast_2d([1.30860E+00,-2.67461E-01,-8.75397E-02]),
            "localopt_method": "ibcdfo_manifold_sampling",
            "run_max_eval": 100 * (n + 1),
            "components": 11,
            "lb": np.array([1.0, -0.75, -0.1]),  # lower bound for input
            "ub": np.array([4.0, 0.00, 0.1]),  # upper bound for input
            "hfun": hfun,
        },
    )

    ensemble.alloc_specs = AllocSpecs(
        alloc_f=alloc_f,
        user={
            "async_return": False,  # False causes batch returns
        },
    )

    # Instruct libEnsemble to exit after this many simulations
    ensemble.exit_criteria = ExitCriteria(sim_max=40)

    # Seed random streams for each worker, particularly for gen_f
    ensemble.add_random_streams()

    # Run ensemble
    ensemble.run()

    if ensemble.is_manager:
        ensemble.save_output(__file__)


if __name__ == "__main__":
    main(sys.argv[1:])
