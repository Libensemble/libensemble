"""libEnsemble simulator function: CGYRO linear growth rates over a grid of binormal wavenumbers k_j.

For a parameter vector x, ``run_CGYRO_over_KY`` evaluates gamma(x, k_j) for every k_j in
``ky_grid``.  Each (x, k_j) CGYRO run gets its own ``KY_xx`` directory and may execute
concurrently with the others (``kj_concurrency``).  Completed runs are stored in a file-locked
cache, so an interrupted optimization can be restarted without repeating any CGYRO run.
"""

import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path

import jinja2
import numpy as np
from filelock import SoftFileLock as FileLock

from libensemble.message_numbers import MAN_SIGNAL_KILL, TASK_FAILED, WORKER_DONE

# Per-k_j outcome codes. CGYRO_ERROR means CGYRO ran and produced numbers but
# reported a non-convergence error (e.g. "Underflow in calculation of frequency");
# NO_OUTPUT means nothing usable was produced at all. Only the latter indicates a
# broken setup, so only the latter is fatal.
CONVERGED, MAX_TIME, CGYRO_ERROR, NO_OUTPUT = 0, 1, 2, 3
FAILED = NO_OUTPUT  # backwards-compatible alias
DEFAULT_KY_GRID = np.arange(0.15, 0.80, 0.05)  # 13 values, 0.15 ... 0.75
DEFAULT_KAPPA = 1.25222  # elongation, held fixed in the 2D (delta, zeta) studies


def make_max_gamma_over_ky(ky_grid=DEFAULT_KY_GRID, atol=1e-8, rtol=1e-8):
    """Outer function h(z) = max_j z_j / k_j for manifold sampling, built from the same grid the simulator uses.

    Follows the IBCDFO hfun interface: hfun(z) -> (h, grads, hashes); hfun(z, H0) -> (h, grads) for the
    manifolds named in H0.  Defining it here keeps the k grid in one place.
    """
    ky = np.asarray(ky_grid, dtype=float)

    def hfun(z, H0=None):
        z = np.asarray(z, dtype=float).ravel()
        assert z.size == ky.size, f"Expected {ky.size} values in z (got {z.size})."
        vals = z / ky
        if H0 is None:
            h = float(np.max(vals))
            inds = np.where(np.abs(h - vals) <= atol + rtol * np.abs(vals))[0]
            grads = np.zeros((z.size, len(inds)))
            for col, j in enumerate(inds):
                grads[j, col] = 1.0 / ky[j]
            return h, grads, [str(j) for j in inds]
        js = [int(k) for k in H0]
        grads = np.zeros((z.size, len(js)))
        for col, j in enumerate(js):
            grads[j, col] = 1.0 / ky[j]
        return vals[js], grads

    return hfun


def load_cache(path):
    with FileLock(path + ".lock"):
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)
        return np.array([], dtype=object)


def append_cache(path, entry):
    with FileLock(path + ".lock"):
        db = np.load(path, allow_pickle=True) if os.path.exists(path) else np.array([], dtype=object)
        db = np.append(db, entry)
        tmp = f"{path}.tmp.{os.getpid()}"
        np.save(tmp, db)
        os.replace(tmp + ".npy", path)


def find_cached(cache, key):
    for entry in cache:
        v = entry["var_vals"]
        if np.shape(v) == np.shape(key) and np.allclose(v, key, rtol=1e-8, atol=1e-8):
            return entry
    return None


def classify_exit(msg):
    """Map the last line of out.cgyro.info to an outcome code."""
    m = (msg or "").strip().lower()
    if "converged" in m:
        return CONVERGED
    if re.search(r"max[ _]t", m):
        return MAX_TIME
    return CGYRO_ERROR


def read_outputs(run_dir):
    """Return ([omega, gamma], last line of out.cgyro.info) from a finished CGYRO run."""
    freq = np.atleast_2d(np.loadtxt(run_dir / "out.cgyro.freq"))[-1]
    with open(run_dir / "out.cgyro.info") as f:
        lines = [line.strip() for line in f if line.strip()]
    return np.asarray(freq[:2], dtype=float), (lines[-1] if lines else "")


def render_input(template_text, values, dest):
    dest.write_text(jinja2.Template(template_text).render(values))


def _launch(exctr, run, submit_kwargs, parse_cmd):
    # The executor launches tasks in the current directory, so enter the run directory first.
    run["attempts"] += 1
    here = os.getcwd()
    os.chdir(run["dir"])
    try:
        if parse_cmd and run["attempts"] == 1:
            subprocess.run(parse_cmd, shell=True, check=False)
        n = run["attempts"]
        run["task"] = exctr.submit(app_name="cgyro", stdout=f"cgyro_{n}.out", stderr=f"cgyro_{n}.err", **submit_kwargs)
    finally:
        os.chdir(here)
    run["t0"] = time.time()


def _fail(run, note):
    run["fout"] = np.full(2, np.nan)
    run["msg"] = note
    run["status"] = FAILED


def run_CGYRO_over_KY(H, persis_info, sim_specs, libE_info):
    """Evaluate gamma(x, k_j) for all k_j; returns fvec plus per-k_j status, attempts and wall time.

    ``sim_specs["user"]`` keys: input_filename, input_names (order: kappa, x..., KY), cache_path,
    and optionally kappa, ky_grid, kj_concurrency, max_attempts, poll_interval, submit_kwargs,
    parse_cmd, omp_threads.
    """
    u = sim_specs["user"]
    exctr = libE_info["executor"]
    ky_grid = np.asarray(u.get("ky_grid", DEFAULT_KY_GRID), dtype=float)
    x = np.atleast_1d(np.squeeze(H["x"]))
    max_attempts = u.get("max_attempts", 4)
    max_conc = max(1, int(u.get("kj_concurrency", 1)))
    poll_s = u.get("poll_interval", 1.0)
    submit_kwargs = dict(u.get("submit_kwargs", {}))
    parse_cmd = u.get("parse_cmd")
    cache_path = u["cache_path"]
    input_name = u["input_filename"]
    workdir = Path.cwd()
    template = (workdir / input_name).read_text()
    if u.get("omp_threads"):
        os.environ["OMP_NUM_THREADS"] = str(u["omp_threads"])

    runs = [
        {"ky": ky, "key": np.hstack([u.get("kappa", DEFAULT_KAPPA), x, ky]), "dir": workdir / f"KY_{ky:.2f}",
         "attempts": 0, "wall": 0.0, "fout": None, "msg": "", "status": None, "cached": False}
        for ky in ky_grid
    ]

    cache = load_cache(cache_path)
    todo = deque()
    for i, run in enumerate(runs):
        hit = find_cached(cache, run["key"])
        if hit is not None and not np.all(np.isfinite(np.asarray(hit["obj_vals"], dtype=float))):
            hit = None  # stale non-finite entry from an older run; recompute it
        if hit is not None:
            run["fout"], run["msg"] = np.asarray(hit["obj_vals"], dtype=float), hit["string_out"]
            run["status"], run["cached"] = classify_exit(run["msg"]), True
        else:
            run["dir"].mkdir(exist_ok=True)
            values = dict(zip(u["input_names"], run["key"]))
            render_input(template, values, run["dir"] / input_name)
            todo.append(i)

    running, killed = {}, False
    while (todo or running) and not killed:
        while todo and len(running) < max_conc:
            i = todo.popleft()
            try:
                _launch(exctr, runs[i], submit_kwargs, parse_cmd)
                running[i] = runs[i]["task"]
            except Exception as err:  # launch problems must not take down the worker
                _fail(runs[i], f"launch error: {err}")
        if not running:
            continue
        time.sleep(poll_s)
        if exctr.manager_kill_received():
            for task in running.values():
                exctr.kill(task)
            killed = True
            break
        for i, task in list(running.items()):
            task.poll()
            if not task.finished:
                continue
            del running[i]
            run = runs[i]
            run["wall"] += time.time() - run["t0"]
            try:
                run["fout"], run["msg"] = read_outputs(run["dir"])
            except Exception as err:  # no usable output: flag it, and do not cache it
                _fail(run, f"unreadable output: {err}")
                continue
            run["status"] = classify_exit(run["msg"])
            if run["status"] == MAX_TIME and run["attempts"] < max_attempts:
                todo.appendleft(i)
                continue
            append_cache(cache_path, {"obj_vals": run["fout"], "var_vals": run["key"], "string_out": run["msg"]})

    for run in runs:
        if not run["cached"] and run["status"] is not None:
            # x and k are recorded here too: later cost/failure analyses read these
            # directories directly, without the cache or the libEnsemble history.
            (run["dir"] / "run_status.txt").write_text(
                f"status = {run['status']}\nattempts = {run['attempts']}\nwall_s = {run['wall']:.1f}\n"
                f"exit = {run['msg']}\nky = {run['ky']:.5f}\n"
                f"x = {np.array2string(np.asarray(run['key'], dtype=float), precision=8)}\n"
            )
        if run["status"] is None:
            _fail(run, "not run (worker killed)")

    fvec = np.array([r["fout"][1] for r in runs])
    statuses = [r["status"] for r in runs]

    # Producing no output at all points at a broken setup rather than a physics
    # result, and an all-NaN fvec would let the job "succeed" while feeding the
    # optimizer garbage. A CGYRO error is different: the run executed and returned
    # numbers, so it is flagged and kept, exactly as the earlier serial code did.
    if not killed and all(s == NO_OUTPUT for s in statuses):
        raise RuntimeError(
            f"no usable output from any of the {len(runs)} CGYRO runs for "
            f"x={np.asarray(x).tolist()}; first reason: {runs[0]['msg']}"
        )
    # An underflow returns gamma = 0, and because the objective is max_j gamma_j/k_j
    # a zero can only ever lower it -- so a numerical failure can masquerade as a
    # good point. Optimization runs therefore stop on one; sampling runs (maps) keep
    # the value and carry on, which is what the earlier serial code did.
    errored = [r for r in runs if r["status"] == CGYRO_ERROR]
    if errored and u.get("stop_on_cgyro_error", False):
        detail = "; ".join(f"k={r['ky']:.2f}: {r['msg']}" for r in errored[:3])
        raise RuntimeError(
            f"CGYRO reported an error at {len(errored)} of {len(runs)} wavenumbers for "
            f"x={np.asarray(x).tolist()} (gamma was returned as "
            f"{', '.join(f'{r['fout'][1]:g}' for r in errored[:3])}). Stopping rather than "
            f"letting a numerical failure look like a good point. Details: {detail}"
        )
    for r in errored:
        if not r["cached"]:
            print(f"WARNING: CGYRO error at k={r['ky']:.2f}, value kept and flagged: {r['msg']}", flush=True)
    output = np.zeros(1, dtype=sim_specs["out"])
    output["fvec"][0] = fvec
    output["f"] = np.max(fvec / ky_grid)
    output["convstatement"] = runs[-1]["msg"]
    names = output.dtype.names
    if "kj_status" in names:
        output["kj_status"][0] = statuses
    if "kj_attempts" in names:
        output["kj_attempts"][0] = [r["attempts"] for r in runs]
    if "kj_wall" in names:
        output["kj_wall"][0] = [r["wall"] for r in runs]

    if killed:
        calc_status = MAN_SIGNAL_KILL
    else:
        calc_status = WORKER_DONE if all(s == CONVERGED for s in statuses) else TASK_FAILED
    return output, persis_info, calc_status
