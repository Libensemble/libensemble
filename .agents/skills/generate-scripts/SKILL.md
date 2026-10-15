---
name: generate-scripts
description: Generate libEnsemble calling scripts based on user requirements
---

You are generating libEnsemble scripts. libEnsemble coordinates parallel simulations
with generator-directed optimization or sampling. You will produce a calling script
and, when an external application is involved, a sim function file.

libEnsemble repository: https://github.com/Libensemble/libensemble
If not running inside the libEnsemble repo, find examples and source code there.

## Workflow

1. If converting an existing Xopt or Optimas workflow to libEnsemble, use the
   existing generator and VOCS settings exactly as-is — even if it is a sampling
   or exploration generator. Do not switch to a classic generator unless the user
   specifically asks.
   Otherwise, if there is not a clear generator to use, read `references/generators.md`
   to determine which generator and
   style to use. If a specific generator is identified (e.g., APOSMM), read its
   dedicated guide (e.g., `references/aposmm.md`).

2. Find a relevant example in `libensemble/tests/regression_tests/` and read it as a
   reference. Some examples:
   - Xopt Bayesian optimization (VOCS): `test_xopt_EI_initial_sample.py` — best Xopt
     example as it demonstrates the initial sampling approach Xopt generators need
   - Optimas Ax optimization (VOCS): `test_optimas_ax_sf.py`
   - APOSMM with NLopt (classic): `test_persistent_aposmm_nlopt.py`
   - Random uniform sampling (classic): `test_1d_sampling.py`
   Use glob and grep to find others matching the generator or pattern needed.
   The regression tests have clear descriptions in the docstring.

3. Write the calling script adapting the example to the user's requirements.
   Do not copy test boilerplate from examples
   (e.g., "Execute via one of the following commands..." headers). Set nworkers
   directly in the script (in LibeSpecs) — do not use parse_args or command-line
   arguments unless the user asks for that. If parse_args is not used and no
   options are taken, then do not ever suggest running with "-n/nworkers" or comms.
   Those optins are used only with parse_args (used in tests).

4. If the user has an external application (executable), also write a sim function file
   that uses the executor to run it.

5. If the user provides an input file, check whether it has Jinja2 template markers
   (`{{ varname }}`). If not, create a templated copy: replace parameter values with
   `{{ name }}` markers matching `input_names` in sim_specs (case-sensitive). The sim
   function uses `jinja2.Template` to render the file before each simulation. Never
   modify the user's original file.

6. Verify the scripts:
   - Bounds and dimension match the user's request
   - Executable path is correct
   - For VOCS: variable names are consistent between VOCS definition and sim function
   - For APOSMM: gen_specs outputs include all required fields
   - Input file template markers match input_names (case-sensitive)
   - The app_name in submit() matches register_app()

7. Present a concise summary highlighting: generator choice, bounds, parameters,
   sim_max, and objective field. Do NOT suggest `mpirun` or other MPI
   runner (srun, mpiexec, etc.) to launch libEnsemble unless the user explicitly
   asks for MPI-based comms.

8. Ask the user if they want to run the scripts.

9. If running: execute with `python script.py`. Do not use `mpirun` or other MPI
   runner (srun, mpiexec, etc.) to launch libEnsemble unless the user explicitly
   asks for MPI-based comms for distributing workers. This is unrelated to
   MPIExecutor, which workers use to launch simulation applications across nodes
   — libEnsemble manages node allocation.
   If scripts fail, retry if you can see a fix, otherwise stop. After a successful
   run, read `references/results_metadata.md` and
   `references/finding_objectives.md` to interpret the output.

## Generator style

VOCS (gest-api) is the default style. It uses a VOCS object to define variables and
objectives, and a generator object from Xopt or Optimas. Use VOCS unless the user
explicitly asks for the classic style or the generator only exists in classic form
(e.g., APOSMM, persistent_sampling).

## Defaults

- nworkers defaults to 4 unless the user specifies otherwise (or 1 for sequential
  generators like Nelder-Mead)
- All nworkers are available for simulations
- No alloc_specs needed — all allocator options are available as GenSpecs parameters
- Use `async_return=True` in GenSpecs unless there is a reason to use batch returns

## VOCS generators (Xopt / Optimas)

Key patterns:
- Variables named individually in VOCS: `{"x0": [lb, ub], "x1": [lb, ub]}`
- Objectives named in VOCS: `{"f": "MINIMIZE"}`
- GenSpecs uses `generator=`, `vocs=`, `batch_size=`
- SimSpecs uses `vocs=` or `simulator=` for gest-api style sim functions
- No `add_random_streams()` needed
- Xopt generators need `initial_sample_method="uniform"` and `initial_batch_size=`
  for initial evaluated data. Optimas handles its own sampling.

See `references/generators.md` for the full generator selection guide.

## Classic generators

Used only when the generator has no VOCS version or the user explicitly requests it.
- One worker is consumed by the persistent generator
- Requires `add_random_streams()`
- APOSMM: see `references/aposmm.md` for full configuration details

## Sim function patterns

**Inline sim function** (no external app): Takes `(H, persis_info, sim_specs, libE_info)`
and returns `(H_o, persis_info)`. Or for VOCS gest-api style, takes `input_dict: dict`
and returns a dict. See `libensemble/sim_funcs/` for built-in examples.

**Executor-based sim function** (external app): Uses MPIExecutor to run an application.
Pattern:
1. Register app in calling script: `exctr.register_app(full_path=..., app_name=...)`
2. In sim function: get executor from `libE_info["executor"]`, submit with
   `exctr.submit(app_name=...)`, wait with `task.wait()`
3. Read output file to get objective value
4. Set `sim_dirs_make=True` in LibeSpecs
5. If using input file templating, set `sim_dir_copy_files=[input_file]`

## Results interpretation

After a successful run:
- Load the .npy output file with `np.load()`
- Always filter by `sim_ended == True` before analyzing — rows where sim_ended is False
  contain uninitialized values (often zeros) that are NOT real results
- For APOSMM: check rows where `local_min == True` to find identified minima
- Report the count, location, and objective value of minima or best points found
- If the best objective value is exactly 0.0, verify those rows have sim_ended == True
- See `references/results_metadata.md` for full details

## Reference docs (read as needed)

All paths relative to this skill's directory:

- `references/generators.md` — Generator selection guide, VOCS vs classic
- `references/aposmm.md` — APOSMM configuration, optimizer options, tuning
- `references/finding_objectives.md` — Identifying objective fields in results
- `references/results_metadata.md` — Interpreting history array, filtering results

## User request

$ARGUMENTS
