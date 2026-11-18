"""Local test of cgyro_sim.run_CGYRO_over_KY with a stub CGYRO (no CGYRO, GPUs, or ibcdfo needed).

Run:  python test_cgyro_concurrent.py
Checks serial vs concurrent k_j evaluation, cache replay, max_time retries, and failed runs.
"""
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from cgyro_sim import CONVERGED, FAILED, MAX_TIME, load_cache, run_CGYRO_over_KY
from libensemble import Ensemble
from libensemble.executors import Executor
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

HERE = Path(__file__).resolve().parent
NPTS = 2
KY = np.arange(0.15, 0.80, 0.05)


def run_case(workdir, cache, conc, env=None, max_attempts=4):
    os.chdir(workdir)
    for k in ("STUB_SLEEP", "STUB_MAXTIME_KY", "STUB_MAXTIME_ATTEMPTS", "STUB_FAIL_KY"):
        os.environ.pop(k, None)
    os.environ.update(env or {})
    exctr = Executor()
    exctr.register_app(full_path=str(HERE / "stub_cgyro.py"), app_name="cgyro")
    sampling = Ensemble(
        libE_specs=LibeSpecs(
            nworkers=1, comms="local", gen_on_manager=True, sim_dirs_make=True,
            sim_input_dir=str(HERE / "templates" / "fixed_S"), ensemble_dir_path=f"ensemble_{conc}_{time.time_ns()}",
        ),
        sim_specs=SimSpecs(
            sim_f=run_CGYRO_over_KY, inputs=["x"],
            outputs=[("f", float), ("fvec", float, 13), ("convstatement", "U100"),
                     ("kj_status", int, 13), ("kj_attempts", int, 13), ("kj_wall", float, 13)],
            user={"input_filename": "input.cgyro", "input_names": ["KAPPA", "DELTA", "ZETA", "KY"],
                  "cache_path": str(cache), "kj_concurrency": conc, "max_attempts": max_attempts,
                  "poll_interval": 0.1, "submit_kwargs": {"app_args": "0"}},
        ),
        gen_specs=GenSpecs(
            gen_f=uniform_random_sample, outputs=[("x", float, 2)],
            user={"gen_batch_size": NPTS, "lb": np.array([-0.6, -0.1]), "ub": np.array([0.6, 0.25])},
        ),
        exit_criteria=ExitCriteria(sim_max=NPTS),
        executor=exctr,
    )
    sampling.add_random_streams(seed=7)
    t0 = time.time()
    sampling.run()
    return sampling.H[sampling.H["sim_ended"]], time.time() - t0


def check_hfun():
    from cgyro_sim import make_max_gamma_over_ky

    h13 = make_max_gamma_over_ky(KY)
    z = np.random.default_rng(0).uniform(0.05, 0.4, 13)
    h, grads, hashes = h13(z)
    assert np.isclose(h, (z / KY).max()) and grads.shape == (13, len(hashes))
    h_tie, g_tie, hs_tie = h13(KY * 0.5)  # all ratios equal -> every manifold is active
    assert len(hs_tie) == 13 and np.allclose(np.diag(g_tie), 1 / KY)
    try:
        from ibcdfo.manifold_sampling.h_examples import max_gamma_over_KY
    except ImportError:
        print("hfun self-checks ok (ibcdfo not available for comparison)")
        return
    ky11 = np.linspace(0.10, 0.60, 11)  # the grid hard-coded in that ibcdfo version
    ours = make_max_gamma_over_ky(ky11)
    for z in np.random.default_rng(1).uniform(0.05, 0.4, (20, 11)):
        a, b = ours(z), max_gamma_over_KY(z)
        assert np.isclose(a[0], b[0]) and np.allclose(a[1], b[1]) and a[2] == b[2]
        assert np.allclose(ours(z, H0=["3", "7"])[0], max_gamma_over_KY(z, H0=["3", "7"])[0])
    print("hfun matches ibcdfo.max_gamma_over_KY on its 11-point grid")


def main():
    check_hfun()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = {"STUB_SLEEP": "0.4"}

        Hs, t_serial = run_case(tmp, tmp / "cache_serial.npy", 1, env)
        Hc, t_conc = run_case(tmp, tmp / "cache_conc.npy", 13, env)
        assert np.allclose(Hs["fvec"], Hc["fvec"]), "concurrent fvec differs from serial"
        assert (Hc["kj_status"] == CONVERGED).all() and (Hc["kj_attempts"] == 1).all()
        assert np.allclose(Hc["f"], (Hc["fvec"] / KY).max(axis=1)), "f must be max_j gamma_j/k_j"
        assert t_conc < t_serial / 3, f"no speedup: serial {t_serial:.1f}s, concurrent {t_conc:.1f}s"
        print(f"serial {t_serial:.1f}s   concurrent(13) {t_conc:.1f}s   speedup {t_serial / t_conc:.1f}x")

        Hr, t_replay = run_case(tmp, tmp / "cache_conc.npy", 13, env)
        assert np.allclose(Hr["fvec"], Hc["fvec"]) and (Hr["kj_attempts"] == 0).all()
        print(f"cache replay {t_replay:.1f}s (no CGYRO runs)")

        j = int(np.argmin(abs(KY - 0.30)))
        Hm, _ = run_case(tmp, tmp / "cache_retry.npy", 13, {**env, "STUB_MAXTIME_KY": "0.30", "STUB_MAXTIME_ATTEMPTS": "2"})
        assert (Hm["kj_status"][:, j] == CONVERGED).all() and (Hm["kj_attempts"][:, j] == 3).all()
        assert np.allclose(Hm["fvec"], Hc["fvec"]), "retry must reproduce the converged value"
        print("max_time retry: converged on attempt 3")

        Hx, _ = run_case(tmp, tmp / "cache_exhaust.npy", 13, {**env, "STUB_MAXTIME_KY": "0.30", "STUB_MAXTIME_ATTEMPTS": "9"})
        assert (Hx["kj_status"][:, j] == MAX_TIME).all() and (Hx["kj_attempts"][:, j] == 4).all()
        assert np.isfinite(Hx["fvec"]).all()
        print("max_time exhausted: flagged, last-iterate value kept")

        jf = int(np.argmin(abs(KY - 0.45)))
        Hf, _ = run_case(tmp, tmp / "cache_fail.npy", 4, {**env, "STUB_FAIL_KY": "0.45"})
        assert (Hf["kj_status"][:, jf] == FAILED).all() and np.isnan(Hf["fvec"][:, jf]).all()
        assert len(load_cache(str(tmp / "cache_fail.npy"))) == NPTS * 12, "failed runs must not be cached"
        print("failed run: flagged NaN, not cached; concurrency 4 in waves ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
