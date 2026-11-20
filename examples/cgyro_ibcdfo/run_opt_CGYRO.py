#!/usr/bin/env python
"""Minimize G(x) = max_j gamma(x, k_j)/k_j over CGYRO growth rates with APOSMM + IBCDFO manifold sampling.

Perlmutter, 13 CGYRO runs per evaluation at once (needs a 13-node allocation; Slurm places the steps):
    srun -n 1 python run_opt_CGYRO.py --case fixed_S --nworkers 1 --kj-concurrency 13 \\
        --cache $SCRATCH/cgyro_cache_fixed_S.npy --cgyro-path /path/to/cgyro/src/cgyro --wrapper /path/to/wrap.PERLMUTTER_GPU

Local test with the stub (needs libensemble and ibcdfo, no CGYRO):
    python run_opt_CGYRO.py --case fixed_S --nworkers 1 --executor basic --cgyro-path stub_cgyro.py \\
        --cache stub_cache.npy --kj-concurrency 13 --sim-max 12

Options not listed in --help (--nworkers, --comms, ...) are libEnsemble's own.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CASES = {
    # fixed_S: shaping shears s_delta = s_zeta = 0.  variable_S: shears tied to delta and zeta.
    "fixed_S": {"template": "fixed_S", "lb": [-0.60, -0.10], "ub": [0.60, 0.25]},
    "variable_S": {"template": "variable_S", "lb": [-0.60, -0.10], "ub": [0.60, 0.25]},
}


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False, allow_abbrev=False)
    p.add_argument("-h", "--help", action="store_true")
    p.add_argument("--case", choices=sorted(CASES), default="fixed_S")
    p.add_argument("--template-dir", help="directory containing input.cgyro (default: templates/<case>)")
    p.add_argument("--cache", required=True, help="path of the (x, k_j) result cache; reused across restarts")
    p.add_argument("--sim-max", type=int, default=36, help="stop after this many evaluations")
    p.add_argument("--executor", choices=["mpi", "basic"], default="mpi")
    p.add_argument("--cgyro-path", required=True, help="CGYRO executable (or the stub)")
    p.add_argument("--wrapper", help="launch wrapper placed before the executable (e.g. wrap.PERLMUTTER_GPU)")
    p.add_argument("--parse-cmd", help="command run once per k_j directory before CGYRO (e.g. cgyro_parse.py)")
    p.add_argument("--start", help="starting point as comma-separated values, e.g. 0.2,0.0 (default: all zeros)")
    p.add_argument("--lb", help="override the case's lower bounds, e.g. -0.4,-0.10")
    p.add_argument("--ub", help="override the case's upper bounds, e.g. 0.4,0.25")
    p.add_argument("--kj-concurrency", type=int, default=1, help="number of k_j runs executing at once per evaluation")
    p.add_argument("--ky-min", type=float, default=0.15, help="lowest wavenumber in the objective")
    p.add_argument("--ky-max", type=float, default=0.75, help="highest wavenumber in the objective")
    p.add_argument("--max-attempts", type=int, default=4, help="attempts per k_j when CGYRO stops at max time")
    p.add_argument("--nmpi", type=int, default=4, help="MPI ranks per CGYRO run")
    p.add_argument("--nomp", type=int, default=4, help="OpenMP threads per rank")
    p.add_argument("--nodes-per-run", type=int, default=1)
    p.add_argument("--gpus-per-run", type=int, default=4)
    # "--start -0.3,0.1" looks like a flag to argparse. Join it to the option so a
    # negative starting point works without the user having to write "--start=...".
    src = sys.argv[1:]
    argv = []
    i = 0
    while i < len(src):
        if src[i] in ("--start", "--lb", "--ub") and i + 1 < len(src) and src[i + 1].startswith("-"):
            argv.append(f"{src[i]}={src[i + 1]}")
            i += 2
        else:
            argv.append(src[i])
            i += 1
    known, rest = p.parse_known_args(argv)
    if known.help:
        p.print_help()
        sys.exit(0)
    sys.argv = [sys.argv[0]] + rest  # leave the remainder to libEnsemble
    return known


def main():
    args = parse_cli()
    case = CASES[args.case]
    template_dir = Path(args.template_dir) if args.template_dir else HERE / "templates" / case["template"]

    # Validate the inputs before the heavy imports, so a typo fails instantly.
    n = len(case["lb"])
    lb = np.array([float(v) for v in args.lb.split(",")]) if args.lb else np.array(case["lb"])
    ub = np.array([float(v) for v in args.ub.split(",")]) if args.ub else np.array(case["ub"])
    if lb.size != n or ub.size != n:
        raise SystemExit(f"--lb/--ub need {n} comma-separated values for case {args.case}")
    if np.any(lb >= ub):
        raise SystemExit(f"--lb {lb.tolist()} must be below --ub {ub.tolist()} in every coordinate")
    start = np.array([float(v) for v in args.start.split(",")]) if args.start else np.zeros(n)
    if start.size != n:
        raise SystemExit(f"--start needs {n} comma-separated values for case {args.case}, got {start.size}")
    if np.any(start < lb) or np.any(start > ub):
        raise SystemExit(f"--start {start.tolist()} is outside the bounds {lb.tolist()}..{ub.tolist()}")
    ky_grid = np.round(np.arange(args.ky_min, args.ky_max + 1e-9, 0.05), 10)
    print(f"objective uses {len(ky_grid)} wavenumbers: {ky_grid[0]:.2f}..{ky_grid[-1]:.2f}")
    print(f"starting point: {start.tolist()}   bounds: {lb.tolist()}..{ub.tolist()}")

    import libensemble.gen_funcs

    libensemble.gen_funcs.rc.aposmm_optimizers = "ibcdfo_manifold_sampling"

    from cgyro_sim import make_max_gamma_over_ky, run_CGYRO_over_KY
    from libensemble import Ensemble
    from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
    from libensemble.executors import Executor, MPIExecutor
    from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
    from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

    # With concurrent k_j runs, libEnsemble's node pinning would put every run on the worker's first node;
    # let Slurm place the job steps within the allocation instead.
    # Not conditioned on the executor, so a local stub run with --kj-concurrency > 1
    # exercises exactly the configuration used on the cluster.
    concurrent_kj = args.kj_concurrency > 1

    if args.executor == "mpi":
        exctr = MPIExecutor()
        precedent = args.wrapper or ""
        exctr.register_app(full_path=args.cgyro_path, app_name="cgyro", precedent=precedent)
        extra_args = f"--cpu_bind=cores,verbose -n {args.nmpi * args.nodes_per_run} -c {args.nomp}"
        submit_kwargs = {
            "app_args": "0",
            "procs_per_node": args.nmpi,
            "num_nodes": args.nodes_per_run,
            "extra_args": extra_args,
        }
        if concurrent_kj:
            # num_gpus routes through MPIRunner._assign_gpus, which dereferences
            # resources.worker_resources with no None check -- so it cannot be used
            # while the resource manager is disabled. Ask Slurm for the GPUs directly.
            submit_kwargs["extra_args"] = f"{extra_args} --gpus-per-node {args.gpus_per_run}"
        else:
            submit_kwargs["num_gpus"] = args.gpus_per_run
    else:
        exctr = Executor()
        exctr.register_app(full_path=args.cgyro_path, app_name="cgyro")
        submit_kwargs = {"app_args": "0"}

    ensemble = Ensemble(parse_args=True, executor=exctr)
    # One timestamped directory per run, so a restart does not collide with the
    # previous run's output (libEnsemble aborts if the ensemble dir is non-empty).
    from datetime import datetime

    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,
        sim_dirs_make=True,
        ensemble_dir_path="ensemble_" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S"),
        sim_input_dir=str(template_dir),
        platform_specs={"gpu_setting_type": "option_gpus_per_node"},
        disable_resource_manager=concurrent_kj,
        save_every_k_sims=1,
        # show_resource_sets dereferences the resource manager, so it must be off
        # whenever the manager is disabled, or every sim dies with
        # "'NoneType' object has no attribute 'worker_resources'".
        stats_fmt={"show_resource_sets": not concurrent_kj, "task_datetime": True},
    )

    ensemble.sim_specs = SimSpecs(
        sim_f=run_CGYRO_over_KY,
        inputs=["x"],
        outputs=[
            ("f", float),
            ("fvec", float, len(ky_grid)),
            ("convstatement", "U100"),
            ("kj_status", int, len(ky_grid)),
            ("kj_attempts", int, len(ky_grid)),
            ("kj_wall", float, len(ky_grid)),
        ],
        user={
            "input_filename": "input.cgyro",
            "input_names": ["KAPPA", "DELTA", "ZETA", "KY"],
            "cache_path": os.path.abspath(args.cache),
            "kj_concurrency": args.kj_concurrency,
            "max_attempts": args.max_attempts,
            "omp_threads": args.nomp,
            "submit_kwargs": submit_kwargs,
            "parse_cmd": args.parse_cmd,
            "ky_grid": ky_grid,
        },
    )

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
        ("started_run", bool),
    ]
    ensemble.gen_specs = GenSpecs(
        gen_f=gen_f,
        persis_in=["f", "fvec"] + [g[0] for g in gen_out],
        out=gen_out,
        user={
            "initial_sample_size": 1,
            "stop_after_k_runs": 1,
            "max_active_runs": 1,
            "sample_points": np.atleast_2d(start),
            "localopt_method": "ibcdfo_manifold_sampling",
            "run_max_eval": 100 * (n + 1),
            "components": len(ky_grid),
            "lb": lb,
            "ub": ub,
            "hfun": make_max_gamma_over_ky(ky_grid),
        },
    )

    ensemble.alloc_specs = AllocSpecs(alloc_f=alloc_f, user={"async_return": False})
    ensemble.exit_criteria = ExitCriteria(sim_max=args.sim_max)
    ensemble.add_random_streams()
    ensemble.run()

    if ensemble.is_manager:
        # Tag the output with the case, start point and k grid: the default name is
        # built only from eval and worker counts, so two runs that differ in those
        # settings would silently overwrite each other.
        # No dots in the tag: save_output strips a trailing ".15" as a file extension.
        fmt = lambda v: f"{v:g}".replace(".", "p").replace("-", "m")
        tag = (f"{args.case}_start{'_'.join(fmt(v) for v in start)}_ky{fmt(args.ky_min)}"
               f"_d{fmt(lb[0])}to{fmt(ub[0])}")
        ensemble.save_output(f"run_opt_CGYRO_{tag}")


if __name__ == "__main__":
    main()
