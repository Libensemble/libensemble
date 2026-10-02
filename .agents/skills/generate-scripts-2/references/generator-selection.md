# Generator selection

Preserve a named user choice. Otherwise select by algorithmic need, package availability,
and simulator cost. Do not translate an existing Xopt or Optimas workflow to another
algorithm merely to simplify it.

| Goal | Preferred generator | Package | Key behavior |
|---|---|---|---|
| Independent random sampling | `UniformSample` | libEnsemble | No evaluated seed data; reproducible seed |
| Space-filling bounded design | `LatinHypercubeSample` | libEnsemble | Each suggestion is a separate LHS design |
| Evaluate an existing design | `PreloadedSampleGenerator` | libEnsemble | List-of-dicts or structured array; stops on exhaustion |
| Find multiple local minima | `APOSMM` | libEnsemble + local optimizer | Concurrent local runs after initial evaluations |
| Bayesian optimization | Xopt `ExpectedImprovementGenerator` | Xopt | Requires evaluated initial data; supports constrained EI |
| Sequential simplex search | Xopt `NelderMeadGenerator` | Xopt | Treat as sequential with `batch_size=1` |
| Cartesian parameter scan | Optimas `GridSamplingGenerator` | Optimas | Generates a fixed grid |
| Ax optimization/fidelity/tasks | Optimas Ax generators | Optimas/Ax | Specialized single-, multi-fidelity, and multitask designs |
| GP adaptive sampling | `GP_CAM` / `GP_CAM_Covar` | gpCAM | Batch surrogate training/acquisition |
| Custom policy | subclass `gest_api.Generator` | gest-api | Implement `suggest` and `ingest` |

## Default decisions

- Prefer `UniformSample` for an unspecified sampling request.
- Prefer LHS only when the user wants a space-filling design; generate its complete design
  in one request when stratification across the whole sample matters.
- Ask before choosing an optimizer when the user says only "optimize." Determine whether
  they need one optimum, multiple minima, constraints, multiple objectives, noisy outputs,
  derivatives, fidelities, or Bayesian uncertainty.
- Prefer built-in APOSMM for multiple local minima. Do not use a legacy APOSMM function.
- Use `PreloadedSampleGenerator` for user-supplied points; do not pass points as allocator
  state.

## `GenSpecs` controls

```python
gen_specs = GenSpecs(
    generator=generator,
    vocs=vocs,
    initial_batch_size=8,
    batch_size=4,
)
```

- `initial_batch_size`: first number requested; defaults to `batch_size` when zero.
- `batch_size`: normal request size.
- `initial_sample_method`: `"uniform"`, `"latin_hypercube"`, or a sampler object. It
  creates evaluated warm-up data before the main generator is asked for points.
- `async_return`: default `False`. Set `True` only if the generator can ingest one completed
  result at a time after initialization and faster adaptation outweighs batching.

Existing `H0` can warm-start standardized generators when it contains all required VOCS
fields. For a direct generator API, evaluated records can instead be passed to
`generator.ingest(list_of_dicts)` before constructing/running the ensemble. When VOCS has
constants, verify the selected generator emits their values; wrap suggestions to merge
`vocs.constants` when it does not.

## Placement and concurrency

Standardized generators run as a persistent manager-side thread by default. All
`LibeSpecs(nworkers=N)` workers are therefore normally simulation workers. Set
`gen_on_worker=True` only for a specific reason; that consumes a worker.

Sequential generator logic does not automatically imply `nworkers=1`: initial designs or
queued evaluations may still run concurrently. Set `batch_size=1` where algorithmically
required and choose workers according to useful simulator concurrency.

## Compatibility questions

Before emitting a script, verify that the generator supports:

- continuous versus discrete/categorical/vector variables;
- objective count and MINIMIZE/MAXIMIZE direction;
- constraints and constants;
- gradients or vector residuals if required;
- asynchronous or irregular feedback;
- restart data and cancellation behavior.

VOCS describes a problem but does not guarantee that a selected generator implements all
of these features.
