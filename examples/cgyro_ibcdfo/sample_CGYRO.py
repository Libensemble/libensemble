#!/usr/bin/env python
"""Evaluate CGYRO at a fixed list of parameter points, for a chosen set of wavenumbers.

This is the companion to run_opt_CGYRO.py for work that is not an optimization:
topping up an existing result cache with a wavenumber that was not run before,
or filling gaps in a contour map. Points are independent, so the parallelism is
across points (one libEnsemble worker each), not across k.

Top up every point already in a cache with k=0.10, 13 points at a time:

    srun -n 1 python sample_CGYRO.py --nworkers 13 --case fixed_S \\
        --x-from-cache $SCRATCH/cgyro_cache_fixed_S.npy --ky-grid 0.10 \\
        --cache $SCRATCH/cgyro_cache_fixed_S.npy \\
        --cgyro-path /path/to/cgyro --wrapper /path/to/wrap.PERLMUTTER_GPU

Points whose requested k values are all cached already are skipped, so the run
contains only real work and can be restarted freely.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CASES = {
    "fixed_S": {"template": "fixed_S"},
    "variable_S": {"template": "variable_S"},
}


def parse_cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False, allow_abbrev=False,
    )
    p.add_argument("-h", "--help", action="store_true")
    p.add_argument("--case", choices=sorted(CASES), default="fixed_S")
    p.add_argument("--template-dir", help="directory holding input.cgyro (default: templates/<case>)")
    p.add_argument("--cache", required=True, help="result cache to read and extend")
    p.add_argument("--ky-grid", default="0.10",
                   help="comma-separated k values to evaluate, or min:max:step")
    p.add_argument("--x-from-cache", help="take the point list from the distinct (delta, zeta) in this cache")
    p.add_argument("--x-from-csv", help="take the point list from a CSV with delta,zeta per line")
    p.add_argument("--x-grid", help="build a grid: dmin:dmax:nd,zmin:zmax:nz")
    p.add_argument("--max-points", type=int, help="evaluate at most this many points (for a smoke test)")
    p.add_argument("--delta-range", help="only evaluate points with delta in this window, e.g. -0.4,0.4")
    p.add_argument("--zeta-range", help="only evaluate points with zeta in this window, e.g. -0.10,0.25")
    p.add_argument("--dry-run", action="store_true", help="list the work and exit without running libEnsemble")
    p.add_argument("--executor", choices=["mpi", "basic"], default="mpi")
    p.add_argument("--cgyro-path", required=True)
    p.add_argument("--wrapper")
    p.add_argument("--parse-cmd")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--nmpi", type=int, default=4)
    p.add_argument("--nomp", type=int, default=4)
    p.add_argument("--nodes-per-run", type=int, default=1)
    p.add_argument("--gpus-per-run", type=int, default=4)
    # negative values after these options would otherwise look like flags
    src = sys.argv[1:]
    argv = []
    i = 0
    while i < len(src):
        if src[i] in ("--delta-range", "--zeta-range") and i + 1 < len(src) and src[i + 1].startswith("-"):
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


def parse_ky_grid(spec):
    if ":" in spec:
        lo, hi, step = (float(v) for v in spec.split(":"))
        return np.round(np.arange(lo, hi + 1e-9, step), 10)
    return np.array([float(v) for v in spec.split(",")], dtype=float)


def points_from_cache(path):
    """Distinct (delta, zeta) already present in a cache, in a stable order."""
    from cgyro_sim import load_cache

    seen = {}
    for entry in load_cache(path):
        v = np.asarray(entry["var_vals"], dtype=float)
        seen.setdefault(tuple(np.round(v[1:-1], 10)), None)  # strip kappa and k
    return np.array(list(seen), dtype=float)


def points_from_csv(path):
    return np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=float))


def points_from_grid(spec):
    dpart, zpart = spec.split(",")
    dlo, dhi, nd = dpart.split(":")
    zlo, zhi, nz = zpart.split(":")
    D, Z = np.meshgrid(
        np.linspace(float(dlo), float(dhi), int(nd)),
        np.linspace(float(zlo), float(zhi), int(nz)),
        indexing="xy",
    )
    return np.column_stack([D.ravel(), Z.ravel()])


def main():
    args = parse_cli()
    case = CASES[args.case]
    template_dir = Path(args.template_dir) if args.template_dir else HERE / "templates" / case["template"]
    cache_path = os.path.abspath(args.cache)
    ky_grid = parse_ky_grid(args.ky_grid)

    sources = [args.x_from_cache, args.x_from_csv, args.x_grid]
    if sum(s is not None for s in sources) != 1:
        raise SystemExit("give exactly one of --x-from-cache, --x-from-csv, --x-grid")

    from cgyro_sim import DEFAULT_KAPPA, find_cached, load_cache, run_CGYRO_over_KY

    if args.x_from_cache:
        points = points_from_cache(args.x_from_cache)
    elif args.x_from_csv:
        points = points_from_csv(args.x_from_csv)
    else:
        points = points_from_grid(args.x_grid)

    # A cache accumulates every point any earlier run visited, including ones from
    # wider bounds than we now use. Topping those up is wasted compute, and for this
    # case it is exactly where CGYRO underflows, so allow restricting the region.
    def in_window(x, spec, idx):
        if not spec:
            return True
        lo, hi = (float(v) for v in spec.split(","))
        return lo <= x[idx] <= hi

    n_all = len(points)
    points = np.array([x for x in points
                       if in_window(x, args.delta_range, 0) and in_window(x, args.zeta_range, 1)])
    if len(points) < n_all:
        print(f"region filter: {len(points)} of {n_all} points kept "
              f"(delta {args.delta_range or 'any'}, zeta {args.zeta_range or 'any'})")
    if len(points) == 0:
        raise SystemExit("no points left after the region filter")

    # Skip points whose requested k values are already cached, so the run holds only real work.
    cache = load_cache(cache_path) if os.path.exists(cache_path) else []
    todo = []
    for x in points:
        missing = [ky for ky in ky_grid if find_cached(cache, np.hstack([DEFAULT_KAPPA, x, ky])) is None]
        if missing:
            todo.append(x)
    todo = np.array(todo, dtype=float) if todo else np.zeros((0, points.shape[1]))
    n_need = len(todo)
    if args.max_points:
        todo = todo[: args.max_points]

    print(f"case={args.case}  k={np.round(ky_grid, 4).tolist()}  cache={cache_path}")
    print(f"{len(points)} points in the list, {n_need} need work, {len(points) - n_need} already cached")
    if len(todo) < n_need:
        print(f"--max-points: evaluating only the first {len(todo)} of the {n_need}")
    if args.dry_run or len(todo) == 0:
        for x in todo[:20]:
            print("  todo:", np.round(x, 6).tolist())
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    from libensemble import Ensemble
    from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f
    from libensemble.executors import Executor, MPIExecutor
    from libensemble.specs import AllocSpecs, ExitCriteria, LibeSpecs, SimSpecs

    if args.executor == "mpi":
        exctr = MPIExecutor()
        exctr.register_app(full_path=args.cgyro_path, app_name="cgyro", precedent=args.wrapper or "")
        submit_kwargs = {
            "app_args": "0",
            "procs_per_node": args.nmpi,
            "num_nodes": args.nodes_per_run,
            "num_gpus": args.gpus_per_run,
            "extra_args": f"--cpu_bind=cores,verbose -n {args.nmpi * args.nodes_per_run} -c {args.nomp}",
        }
    else:
        exctr = Executor()
        exctr.register_app(full_path=args.cgyro_path, app_name="cgyro")
        submit_kwargs = {"app_args": "0"}

    n = todo.shape[1]
    H0 = np.zeros(len(todo), dtype=[("x", float, (n,)), ("sim_id", int), ("sim_started", bool)])
    H0["x"] = todo
    H0["sim_id"] = np.arange(len(todo))
    H0["sim_started"] = False

    # One timestamped directory per run: reruns are routine here (the queue limit
    # interrupts long batches), and a fixed name makes libEnsemble abort with
    # "Ensemble directory already existed and wasn't empty". Keeping them apart
    # also preserves each run's CGYRO output for later cost analysis.
    from datetime import datetime

    ensemble = Ensemble(parse_args=True, executor=exctr, H0=H0)
    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,
        sim_dirs_make=True,
        ensemble_dir_path="ensemble_" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S"),
        sim_input_dir=str(template_dir),
        platform_specs={"gpu_setting_type": "option_gpus_per_node"},
        save_every_k_sims=10,
        stats_fmt={"show_resource_sets": True, "task_datetime": True},
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
            "cache_path": cache_path,
            "kj_concurrency": 1,  # the parallelism here is across points, not across k
            "max_attempts": args.max_attempts,
            "omp_threads": args.nomp,
            "submit_kwargs": submit_kwargs,
            "parse_cmd": args.parse_cmd,
            "ky_grid": ky_grid,
        },
    )
    ensemble.alloc_specs = AllocSpecs(alloc_f=alloc_f)
    ensemble.exit_criteria = ExitCriteria(sim_max=len(todo))
    ensemble.add_random_streams()
    ensemble.run()

    if ensemble.is_manager:
        H = ensemble.H[ensemble.H["sim_ended"]]
        status = np.asarray(H["kj_status"])
        print(f"evaluated {len(H)} points x {len(ky_grid)} k")
        print(f"  converged {(status == 0).sum()}, hit max time {(status == 1).sum()}, "
              f"CGYRO error {(status == 2).sum()}, no output {(status == 3).sum()}")
        ensemble.save_output(__file__)


if __name__ == "__main__":
    main()
