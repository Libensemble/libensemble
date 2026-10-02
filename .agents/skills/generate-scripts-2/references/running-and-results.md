# Running and results

## Preflight

Before running:

1. Compile generated Python files (`python -m py_compile ...`).
2. Import-check optional packages.
3. Verify executable and input paths.
4. Use a tiny budget or dry-run mode when the simulator/application supports it.
5. Confirm the output directory will not overwrite valued data. Set
   `reuse_output_dir=True` only intentionally.

`Ensemble.ready()` returns `(ready: bool, issues: list[str])`, not a Boolean. It validates
configured state, including an already-known exit criterion, but it cannot inspect future
arguments to `run(sim_max=...)`. Do not write `if ensemble.ready():`, and do not manipulate
private exit-criteria attributes merely to call it. For ordinary generated scripts, rely
on typed-spec validation plus explicit `run(...)` criteria; use `ready()` only when the
workflow already stores a public, non-deprecated criterion supported by its installed
version.

## Execution

For fixed local workers:

```text
python workflow.py
```

Only use `-n`/`--comms` when the script deliberately uses `Ensemble(parse_args=True)`.
Only use MPI to start manager/workers when distributed MPI communications were requested;
this is separate from MPIExecutor launching simulation applications.

`run()` accepts `sim_max`, `gen_max`, `wallclock_max`, and `stop_val=(field, value)` and
returns `(H, persis_info, exit_flag)`. `stop_val` is a lower-threshold check (`field <=
value`); it does not reverse for a MAXIMIZE objective. Use a transformed metric or another
criterion for maximization. Prefer a hard wall-clock limit for expensive production runs
in addition to an evaluation budget.

Exit flags commonly mean:

- `0`: normal completion
- `1`: exception
- `2`: manager wall-clock timeout
- `3`: process outside the MPI communicator

Treat nonzero flags as non-success unless the workflow intentionally expects them.

## Checkpointing and restart

For expensive runs, configure automatic History snapshots in `LibeSpecs`:

```python
LibeSpecs(
    save_every_k_sims=100,
    save_H_on_completion=True,
    save_H_with_date=True,
    H_file_prefix="campaign_history",
)
```

Use unique run directories by default. Set `reuse_output_dir=True` only for a deliberate
resume. Restart with `H0` only after validating VOCS names/dtypes, required outputs, and
generator/package versions. History restart does not necessarily restore every third-party
generator's internal model; use its supported state restoration when required. Keep
checkpoint files from a timed-out run before returning a nonzero process status.

## Result filtering

History can include generated but unevaluated rows. Always begin with:

```python
completed = H[H["sim_ended"]]
```

Then filter for finite objective values:

```python
import numpy as np
valid = completed[np.isfinite(completed["f"])]
```

For multiple required outputs, combine finiteness masks. If the simulator returns a
user-defined status field, require its success value too. A row can have `sim_ended=True`
yet contain NaN from an application/parser failure, so `sim_ended` alone does not establish
a valid objective.

Use `np.argmin` for MINIMIZE and `np.argmax` for MAXIMIZE. For APOSMM minima:

```python
minima = H[H["sim_ended"] & H["local_min"] & np.isfinite(H["f"])]
```

For multiple objectives, do not reduce to a single "best" row without a user-specified
scalarization or Pareto analysis.

## Saving

```python
if ensemble.is_manager:
    ensemble.save_output("run_results", append_attrs=False)
```

`save_output` saves History and persistent information. With `append_attrs=False`, the
provided History basename is preserved; default attribute suffixes otherwise alter it.
Guard custom reporting, plots, and extra writes with `ensemble.is_manager`.

When loading a saved NumPy History:

```python
H = np.load("run_results.npy", allow_pickle=False)
```

Use the actual filename produced by the installed version/options. Do not claim output was
saved if execution failed before `save_output()`.

## Reporting

Report:

- exit flag and completed/valid evaluation counts;
- objective names and directions;
- best valid point for a scalar objective, or APOSMM minima count/locations;
- failed/non-finite count;
- output path actually written;
- wall-clock timeout or generator exhaustion when relevant.

Never interpret default zeros in unevaluated rows as results. Never suppress application
failures merely to produce a best value.
