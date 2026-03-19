#!/usr/bin/env python

import os
import sys, getopt
import subprocess

import jinja2
import numpy as np
from CGYRO_sim import run_CGYRO_over_KY  # Sim func from current dir

from libensemble import Ensemble
from libensemble.executors import MPIExecutor
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f
from libensemble.specs import AllocSpecs, ExitCriteria, LibeSpecs, SimSpecs


def make_H0_grid(lb, ub, n_samp=30):
    """
    Build a *pre-defined* (deterministic) sample pattern in the 2D box [lb, ub].

    Default pattern: a 6x5 grid = 30 points (matches your previous sim_max=30).
    If n_samp != 30, we build a near-square grid with at least n_samp points,
    then truncate to exactly n_samp.

    Returns:
        H0: structured numpy array with fields needed by give_pregenerated_sim_work
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    assert lb.shape == (2,) and ub.shape == (2,)
    assert np.all(ub > lb)

    # Choose a grid that has >= n_samp points
    if n_samp == 30:
        nx, ny = 6, 5
    else:
        nx = int(np.ceil(np.sqrt(n_samp)))
        ny = int(np.ceil(n_samp / nx))

    xs = np.linspace(lb[0], ub[0], nx)
    ys = np.linspace(lb[1], ub[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])[:n_samp]
    # pts = np.array([[0.5, -0.09], [0.6, -0.08], [-0.2, 0.05]])

    H0 = np.zeros(
        len(pts),
        dtype=[
            ("x", float, (2,)),
            ("sim_id", int),
            ("sim_started", bool),
        ],
    )
    H0["x"] = pts
    H0["sim_id"] = np.arange(len(pts), dtype=int)
    H0["sim_started"] = False
    return H0


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
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/kappa_correction_with_KY_Belli_three",
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/kappa_correction_with_KY_jan_26_pt",
        # sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/kappa_correction_with_KY_jan_26_nt",
        sim_input_dir="/global/homes/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/gamma_over_KY_feb_5",
        platform_specs=platform_specs,
        # reuse_output_dir=True,
        save_every_k_sims=1,
        stats_fmt = {"show_resource_sets": True, "task_datetime": True},
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_CGYRO_over_KY,
        inputs=["x"],
        outputs=[("f", float), ("fvec", float, 13), ("convstatement", "U100")],
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

    # pre-generate work in H0, and use pregenerated-work allocator ---
    lb = np.array([-0.60, -0.1], dtype=float)
    ub = np.array([0.60, 0.1], dtype=float)

    # This replaces LHS + sim_max=30; adjust n_samp and/or make_H0_grid as needed
    n_samp = 25
    n_samp = 9
    ensemble.H0 = make_H0_grid(lb, ub, n_samp=n_samp)

    ensemble.alloc_specs = AllocSpecs(alloc_f=alloc_f)
    ensemble.exit_criteria = ExitCriteria(sim_max=len(ensemble.H0))

    # Run ensemble
    ensemble.run()

    if ensemble.is_manager:
        ensemble.save_output(__file__)


if __name__ == "__main__":
    main(sys.argv[1:])
