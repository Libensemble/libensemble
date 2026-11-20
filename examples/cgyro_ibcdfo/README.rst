CGYRO growth-rate optimization with APOSMM + IBCDFO manifold sampling
======================================================================

Minimizes ``G(x) = max_j gamma(x, k_j) / k_j`` for CGYRO linear growth rates ``gamma`` on a grid of 13 binormal
wavenumbers ``k_j = 0.15, 0.20, ..., 0.75``, over magnetic shaping parameters ``x = (delta, zeta)``.

Files
-----

* ``cgyro_sim.py``: simulator function ``run_CGYRO_over_KY``, and ``make_max_gamma_over_ky`` (the outer function ``h``
  built from the same ``k_j`` grid).
* ``run_opt_CGYRO.py``: driver (APOSMM running IBCDFO manifold sampling).
* ``sample_CGYRO.py``: driver for evaluating a *fixed list* of points, used to top up a cache with a wavenumber
  that was not run before, or to fill gaps in a contour map. See "Evaluating a list of points" below.
* ``templates/fixed_S``, ``templates/variable_S``: CGYRO ``input.cgyro`` Jinja2 templates (shaping shears zero, or
  tied to ``delta`` and ``zeta``).
* ``stub_cgyro.py``, ``test_cgyro_concurrent.py``: a stand-in for CGYRO and a test that needs neither CGYRO nor GPUs.
* ``submit_perlmutter.sh``: Perlmutter job script.

How an evaluation works
-----------------------

For each ``x`` the simulator creates one ``KY_xx`` directory per ``k_j``, renders ``input.cgyro`` into it, and launches
CGYRO there. ``gamma`` is the last row of ``out.cgyro.freq``. If CGYRO stops at its maximum time, the run is resubmitted
(up to ``--max-attempts``). The outputs are ``fvec`` (the 13 values of ``gamma``) and, for each ``k_j``, ``kj_status``
(0 converged, 1 stopped at max time after all attempts, 2 no usable output), ``kj_attempts`` and ``kj_wall``.
A run with status 1 keeps its last-iterate ``gamma``. A run with status 2 returns NaN and is not cached.

Completed runs are stored in the file given by ``--cache``, keyed on ``(kappa, x, k_j)``. Rerunning with the same
cache skips every completed CGYRO run, so an optimization can be continued across batch jobs.

Adding a wavenumber to an existing run
--------------------------------------

To re-run an optimization with an extra ``k`` (say adding ``k=0.10`` to a run done on 0.15-0.75), fill the
new ``k`` in **first** with ``sample_CGYRO.py``, then re-run the optimization. Both steps read the same cache,
so the order matters only for speed:

1. ``sample_CGYRO.py --ky-grid 0.10 --x-from-cache <cache>`` evaluates the new ``k`` at every point in the
   cache that lacks it, spread across workers (one point each). Points that already have it are skipped, so
   this picks up exactly the ones the earlier run visited.
2. ``run_opt_CGYRO.py --ky-min 0.10 ...`` then replays those evaluations from the cache in seconds and only
   computes genuinely new points.

Done the other way round, every evaluation serializes on its single missing ``k`` run, because within one
evaluation the concurrency is across ``k`` and there is only one left to do.

Note that the stored history's ``fvec`` width records which grid a run used, and the plotting scripts need
``--history-ky-min`` set to match it.

Starting point
--------------

``--start`` sets where the local optimization begins, as comma-separated values (default: all zeros, which is
what the published runs used)::

    python run_opt_CGYRO.py --case fixed_S --start 0.2,0.0 ...

The value is checked against the case bounds before any heavy import, so a typo fails immediately. Keep the
bounds the same across runs that are meant to be compared; if a path leaves the region covered by the contour
map, extend the map with ``sample_CGYRO.py`` rather than shrinking the bounds.

Concurrency
-----------

``--kj-concurrency N`` runs up to ``N`` of the 13 CGYRO runs for a point at once (default 1, i.e. one after another).
With the MPI executor and ``N > 1``, libEnsemble's resource manager is disabled and Slurm places the ``srun`` steps within
the allocation, so the job needs about ``N`` nodes when each run uses one node. This has been tested with the stub only;
see "Testing" for the Perlmutter check.

Evaluating a list of points
---------------------------

``sample_CGYRO.py`` evaluates points that are given to it rather than chosen by an optimizer. The points are
independent, so here the parallelism is **across points** (one libEnsemble worker each, one node per worker),
not across ``k`` within a point. Supply the list with one of ``--x-from-cache`` (every distinct ``(delta, zeta)``
already in a cache), ``--x-from-csv``, or ``--x-grid``::

    srun -n 1 python sample_CGYRO.py --nworkers 13 --case fixed_S \
        --cache $SCRATCH/cgyro_cache_fixed_S.npy --x-from-cache $SCRATCH/cgyro_cache_fixed_S.npy \
        --ky-grid 0.10 --cgyro-path /path/to/cgyro --wrapper /path/to/wrap.PERLMUTTER_GPU

Points whose requested ``k`` values are already cached are skipped, so a rerun after an interrupted job is a
no-op and costs nothing. Use ``--dry-run`` to see how much work a command would do before submitting it, and
``--max-points`` for a smoke test. ``submit_perlmutter_sample.sh`` wraps this for Slurm.

**Check the environment on a login node before submitting.** The batch job loads a module set that can change
which Python is first on ``PATH``, and the dependencies (``numpy``, ``jinja2``, ``filelock``, ``libensemble``)
must be importable by *that* Python. A ``--dry-run`` needs no compute nodes and catches an import error or a
wrong point count in seconds::

    source /path/to/arash_gx_modules.sh
    python -m pip install --user filelock       # if the check below reports it missing
    python sample_CGYRO.py --dry-run --case fixed_S --ky-grid 0.10 \
        --cache $CACHE --x-from-cache $CACHE --cgyro-path $GACODE_ROOT/cgyro/src/cgyro

The submit script repeats this check and exits before the work starts if anything is missing.

Testing
-------

Local, no CGYRO needed (requires ``libensemble``, ``jinja2``, ``filelock``)::

    python test_cgyro_concurrent.py

Perlmutter smoke test (not yet run): one evaluation with 4 concurrent runs in the debug queue, using a fresh cache::

    sbatch -N 4 --export=ALL,CASE=fixed_S,SIMMAX=1,KJ_CONCURRENCY=4 -q debug -t 00:30:00 submit_perlmutter.sh

The first evaluation is at ``x = (0, 0)``. Its ``fvec`` should agree with the earlier serial run to within CGYRO's
run-to-run variation::

    0.08125 0.12258 0.1668 0.21346 0.25376 0.28506 0.3072 0.32058 0.3263 0.32632 0.32013 0.31318 0.30539

Check ``libE_stats.txt`` and the ``KY_xx/run_status.txt`` files, and confirm in ``sacct -j <jobid> --format=JobID,NodeList,Elapsed``
that the job steps ran on different nodes at the same time.
