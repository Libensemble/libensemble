# libEnsemble Generator Functions

This guide is for choosing a generator when one is not already provided. If the user
is converting an existing workflow that already has a generator, use that generator
as-is — do not use this guide to replace it.

libEnsemble supports two styles of generator configuration:

- **VOCS generators (gest-api)** — The default style. Uses a VOCS object to define variables, objectives, and constraints. The generator is passed as an object from Xopt, Optimas, or another gest-api compatible library.
- **Classic generators** — libEnsemble-native gen functions configured via `gen_f`, explicit `inputs`/`outputs`, and `user` dicts with bounds/parameters. Used only when the generator has no VOCS version or the user explicitly requests it.

## When to Choose a Generator Style

**VOCS is the default style.** Any generator from Xopt or Optimas is always VOCS — these libraries provide many generators covering optimization, sampling, surrogate modeling, and more. Do not switch an Xopt or Optimas generator to a classic libEnsemble generator.

Use **classic generators** only when:
- The user explicitly asks for the classic/traditional style
- The generator does not have a VOCS version (APOSMM, persistent_sampling)

## Choosing a generator

| Goal | Suggested generator | Style | Package |
|------|---------------------|-------|---------|
| Bayesian optimization | Xopt (e.g., Expected Improvement) | VOCS | `xopt` |
| Sampling / exploration | Xopt (e.g., Latin Hypercube) | VOCS | `xopt` |
| Ax-based optimization, multi-fidelity, multi-task | Optimas | VOCS | `optimas` |
| Simplex optimization | Xopt Nelder-Mead | VOCS | `xopt` |
| Multi-objective Bayesian | Xopt MOBO | VOCS | `xopt` |
| GP-based adaptive sampling | gpCAM | VOCS or Classic | `gen_classes/gpCAM` |
| Find multiple local minima | APOSMM | VOCS or Classic | `gen_classes/aposmm` |
| Random/uniform sampling | Sampling | VOCS or Classic | `gen_classes/sampling` |

Xopt and Optimas each provide many generators beyond those listed here. If the
generator choice is not clear, check the library documentation:
- Xopt: https://github.com/xopt-org/Xopt — algorithms at https://xopt.xopt.org/algorithms/
- Optimas: https://github.com/optimas-org/optimas

If the user says "optimize" without specifics -> Xopt (VOCS).
If the user says "Xopt", "VOCS", "Optimas", or names a specific generator from those libraries -> VOCS style.
If the user says "Ax", "multi-fidelity", "multi-task" -> Optimas (VOCS).
If the user says "find minima", "multiple local minima" -> APOSMM (classic).
If the user says "sample", "explore", "sweep" -> Xopt or Optimas can do this (VOCS), or persistent sampling (classic).

## VOCS Generators (gest-api)

VOCS is the default configuration style for generators in libEnsemble. Configuration uses a VOCS object to define the optimization problem and a generator object. Generators may come from Xopt, Optimas, libEnsemble, or other gest-api compatible libraries.

### Key patterns

- Variables are named individually in VOCS (`{"x0": [lb, ub], "x1": [lb, ub]}`)
- Objectives are named in VOCS (`{"f": "MINIMIZE"}`)
- GenSpecs uses `generator=`, `vocs=`, and `batch_size=`
- SimSpecs uses `vocs=` or `simulator=` for gest-api style sim functions
- No alloc_specs needed (default is correct)
- No `add_random_streams()` needed
- Use `async_return=True` in GenSpecs unless the generator requires batch returns

### Initial sampling

Some generators require evaluated data before they can suggest points. Set `initial_sample_method` in GenSpecs to have libEnsemble produce and evaluate an initial sample before starting the generator:

- `initial_sample_method="uniform"` — uniform random sample from VOCS bounds
- `initial_batch_size` — required, specifies how many sample points to produce

Generators that handle their own sampling do not need this.

### Sim function adaptation

When using VOCS generators with an executor-based sim function, the sim must read individual variable names from H rather than unpacking `H["x"]`. The `input_names` in `sim_specs["user"]` should match the VOCS variable names directly.

## Classic Generators

### persistent_sampling (persistent_uniform)
Random uniform sampling across parameter space. After the initial batch, creates p new random points for every p points returned.

gen_specs["user"]: `lb`, `ub`, `initial_batch_size`
gen_specs outputs: `x (float, n)`

### APOSMM (persistent_aposmm)
See `reference_docs/aposmm.md` for full details.
Asynchronously Parallel Optimization Solver for finding Multiple Minima.
