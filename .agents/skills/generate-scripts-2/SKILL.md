---
name: generate-scripts-2
description: Generate self-contained, modern libEnsemble 2.x workflows for sampling, optimization, external applications, and HPC resources
---

Generate runnable libEnsemble 2.x scripts from the user's requirements. Resolve every
`references/...` path relative to this loaded `SKILL.md`; never assume the skill lives at a
specific `.claude`, `.agents`, or repository path. Use only the standardized gest-api/VOCS
interfaces described here. Do not depend on access to the
libEnsemble repository and do not emit legacy `libE()`, `gen_f`, `sim_f`, bare-spec
dictionary, or explicit allocation-function patterns.

## Workflow

1. Read `references/intake-and-validation.md`. Extract the problem definition,
   termination budget, parallelism, generator intent, simulator interface, dependencies,
   files, and resources. Ask only questions whose answers materially change the script.
   Never invent bounds, objective direction, executable paths, output parsing, or HPC
   resource requirements. Missing values block execution, not necessarily generation:
   produce a clearly marked scaffold when its structure is still useful and safe.

2. Read `references/generator-selection.md`. Preserve a user-specified gest-api generator
   and its VOCS. Otherwise choose the least-complex suitable generator, preferring
   libEnsemble's built-in sampling, APOSMM, or preloaded-sample classes before optional
   ecosystems. Read `references/aposmm.md` or `references/external-generators.md` when
   applicable. Check that every non-core package is installed before relying on it.

3. Read `references/canonical-patterns.md`. For a basic Python simulator, adapt
   `examples/local_sampling.py`. For an existing set of points, adapt
   `examples/preloaded_points.py`. Keep the complete script structure: `Ensemble`, VOCS,
   typed `SimSpecs`/`GenSpecs`/`LibeSpecs`, `generator=` and `simulator=`, a main guard,
   explicit `ensemble.run(...)` stopping criteria, and manager-only result reporting/saving.

4. For an executable, read `references/executors-and-files.md`. Use `Executor` for a
   normal subprocess and `MPIExecutor` only when the application itself needs an MPI or
   resource-aware launcher. Keep application launch separate from manager/worker launch.
   Use isolated simulation directories for file-producing applications. Never modify an
   original user input file; copy or render it into each simulation directory.

5. For clusters, GPUs, variable task sizes, or scheduler scripts, also read
   `references/hpc-resources.md`. Do not guess machine topology, launcher, scheduler
   directives, process counts, GPU counts, or platform settings.

6. Validate the generated files against `references/intake-and-validation.md`. In
   particular, verify exact names across VOCS, simulator inputs/returns, generator
   mappings, executable registration/submission, parser output, and result analysis.
   Run a syntax/import check when tools are available. For built-in adapter generators,
   also reject ambiguous field mappings such as a scalar VOCS variable named `x` in a
   multi-variable problem unless an explicit tested mapping resolves the collision.

7. Summarize generated files, generator choice, variables/bounds, objectives and
   directions, batch size, workers, stopping criteria, dependencies, and application
   resources. Identify every placeholder the user must replace. For production artifacts,
   include a minimal dependency manifest with tested versions when the user wants one.

8. Ask before executing a generated workflow. For a fixed local script run
   `python script.py`; do not append `-n`, `--comms`, or use `mpirun`/`srun` unless the
   script intentionally uses `Ensemble(parse_args=True)` and the user requested that
   launch mode. `MPIExecutor` may launch simulation applications with MPI even when the
   calling script itself runs with plain Python.

9. If execution is approved, read `references/running-and-results.md`, run the smallest
   useful validation first, fix actionable failures, and report only completed, finite
   results. Do not claim a `.npy` result exists unless `save_output()` ran successfully.
   Treat an exception raised from `ensemble.run()` separately: manager-only code after the
   call will not execute, though libEnsemble may write its own abort checkpoint.

## Non-negotiable defaults

- Generate modern standardized interfaces only. If the request requires a legacy-only
  feature, explain that this skill does not generate it and ask whether a modern design
  is acceptable.
- Prefer programmatic `LibeSpecs(nworkers=...)`; use `parse_args=True` only when requested.
- Standardized generators run on manager Worker 0 by default, so all `nworkers` are
  normally available for simulations. Do not subtract a generator worker unless setting
  `gen_on_worker=True` intentionally.
- Omit `AllocSpecs` unless a documented modern requirement cannot be represented through
  `GenSpecs`/`LibeSpecs`.
- Do not set `async_return=True` globally. Choose it only when the generator supports
  one-at-a-time feedback after initialization.
- Use `safe_mode=True` for generated production workflows unless a documented requirement
  needs protected-field writes. Never return protected History metadata from a simulator.
- Use a reproducible seed where the selected generator supports one.
- Keep examples' numbers and paths out of user scripts unless they match the request.

## References

Read only what the request needs; all paths are relative to this skill directory.

- `references/intake-and-validation.md` — requirements and final checks
- `examples/local_sampling.py` — runnable built-in sampling workflow
- `examples/preloaded_points.py` — runnable workflow for evaluating supplied points
- `references/canonical-patterns.md` — adaptation guidance and generator/VOCS pitfalls
- `references/generator-selection.md` — generator decision table and batch behavior
- `references/aposmm.md` — standardized APOSMM configuration
- `references/external-generators.md` — Xopt, Optimas, and gpCAM
- `references/executors-and-files.md` — subprocesses, MPI applications, files, failures
- `references/hpc-resources.md` — schedulers, resource sets, GPUs, platform settings
- `references/running-and-results.md` — validation, execution, saving, interpretation

## User request

$ARGUMENTS
