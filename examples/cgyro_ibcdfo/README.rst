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
