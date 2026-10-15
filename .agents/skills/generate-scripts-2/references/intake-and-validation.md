# Intake and validation

## Ask for missing requirements

Ask only for information that cannot be inferred safely.

### Problem

- Variable names, types, shapes, and bounds or categories
- Objective/observable/constraint names and objective directions
- Constraint relation and threshold
- Constants passed to every simulation
- Sampling, one optimum, multiple minima, Bayesian optimization, grid scan, or preloaded points
- Seed/restart data and reproducibility seed

### Execution

- Python callable or external executable
- Executable path and whether it is serial, threaded, GPU-enabled, or MPI
- Exact command-line arguments, input files, and output parser
- Processes, nodes, cores, GPUs, and timeout per simulation
- Files/directories to copy or symlink and desired output directory
- Local machine or scheduler/platform details

### Ensemble

- `nworkers` (default to 4 only for a generic local workflow)
- Generator batch size and whether partial asynchronous feedback is supported
- At least one stopping criterion: `sim_max`, `gen_max`, `wallclock_max`, or `stop_val`
- Whether to save History and the output basename
- Whether the script should use fixed settings or `Ensemble(parse_args=True)`

## Name contract

Treat VOCS as the schema shared by generation, simulation, and results.

- Every VOCS variable read by the simulator must use the exact same spelling and case.
- A dict simulator must return every objective and constraint needed by the generator.
  Validate required keys, numeric types/shapes, and finiteness before returning; a missing key
  may otherwise remain a plausible zero in History.
- Returned objective, observable, and constraint values must be scalar numeric values unless
  the VOCS explicitly defines another dtype/shape.
- VOCS constants are not necessarily populated by every generator. Ensure every suggestion
  includes the declared constant values (or wrap the generator to add them), then verify the
  simulator receives them. Do not generate constants as variables.
- `MINIMIZE` and `MAXIMIZE` are not interchangeable.
- For `LibensembleGenerator` adapters (including built-in `UniformSample` and
  `LatinHypercubeSample`), internal `x` normally maps to all VOCS variables. If a
  multi-variable VOCS itself contains a scalar variable named `x`, automatic mapping can
  collide with that field and produce shape errors. Rename that History/VOCS field (for
  example `x` to `x0`) and explicitly tell the user. Do not invent an ad hoc pure generator
  merely to preserve the name, and do not assume `variables_mapping` can make one History
  field simultaneously scalar and vector. Never emit the ambiguous case unchecked.
- For adapter-based generators, verify every `variables_mapping` target and order.
- Avoid protected History names such as `sim_id`, `sim_started`, `sim_ended`, `sim_worker`,
  `gen_worker`, `gen_informed`, timing fields, and `kill_sent` for user outputs.

## Dependency check

The base workflow requires `libensemble`, `gest-api`, and NumPy. Never assume optional
packages are installed. Check imports before generation or clearly provide installation
requirements for Xopt, Optimas, gpCAM, SciPy/NLopt optimizer backends, Jinja2, or application
libraries. Do not silently replace a requested generator because its package is missing.

## Final checklist

- Script has an `if __name__ == "__main__":` guard.
- Uses `Ensemble`, typed specs, VOCS, `generator=`, and `simulator=`.
- No legacy `libE()`, `gen_f`, `sim_f`, bare spec dictionaries, or custom allocator.
- Bounds, categories, dimensions, constants, constraints, and objective directions match.
- Generator can handle the requested variable types and constraints.
- `initial_batch_size`, initial sampling, and seed data satisfy generator requirements.
- `batch_size <= nworkers` unless queued generation is intentional.
- `async_return` is omitted unless explicitly justified.
- At least one run stopping criterion is present.
- Fixed-worker scripts do not advertise `-n` or `--comms` CLI arguments.
- The registered `app_name` exactly matches `submit(app_name=...)`.
- Executable and copied input paths exist or are conspicuous placeholders.
- External app arguments and parser agree with the application's real interface. Validate
  the exact token list produced by plain whitespace splitting; quotes in `app_args` are not
  removed or honored.
- Every simulation has isolated files when concurrent tasks write fixed filenames.
- Timeout handling terminates the application; failures are not reported as valid minima.
- MPI process/GPU requests fit each worker's assigned resource sets.
- Postprocessing runs only on the manager and filters `sim_ended` plus finite values.
- `save_output()` is present before promising a saved History file. For a returned nonzero
  exit flag, save before raising/reporting it. For exceptions raised inside `run()`, code
  after `run()` cannot save; rely on `save_H_and_persis_on_abort=True` or choose a compatible
  fault-tolerant simulator policy, and describe the resulting filename behavior accurately.
- Syntax and imports are checked when an environment is available.

## Placeholder policy

Use loud placeholders such as `Path("/REPLACE/WITH/APP")` or raise a clear error rather
than inventing an executable path, parser, scheduler account, or resource count. Summarize
all placeholders after generating files.
