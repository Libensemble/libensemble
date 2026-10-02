# Optional generator ecosystems

Preserve an existing Xopt, Optimas, or gpCAM generator and VOCS unless the user asks to
change algorithms. These packages are not guaranteed by a base libEnsemble installation;
check their imports and version-specific constructor APIs.

## Xopt

### Expected improvement

```python
from xopt.generators.bayesian.expected_improvement import ExpectedImprovementGenerator

generator = ExpectedImprovementGenerator(vocs=vocs)
gen_specs = GenSpecs(
    generator=generator,
    vocs=vocs,
    initial_batch_size=8,
    initial_sample_method="uniform",
    batch_size=4,
)
```

Expected improvement requires evaluated training data. Use exactly one of:

- libEnsemble warm-up through `initial_sample_method` and `initial_batch_size`;
- `generator.ingest([...])` with evaluated dictionaries before the run;
- valid warm-start History (`H0`).

`initial_sample_method` also accepts `"latin_hypercube"` or a sampler object. Do not copy
an explicit allocator from transitional tests. Keep `async_return=False` unless the
installed Xopt generator is known to tolerate irregular one-at-a-time updates.

Xopt EI can use VOCS constraints and constants when supported by the selected generator.
The simulator must return every objective and constraint named in VOCS.

### Sequential Nelder-Mead

```python
from xopt.generators.sequential.neldermead import NelderMeadGenerator

generator = NelderMeadGenerator(vocs=vocs)
gen_specs = GenSpecs(generator=generator, vocs=vocs, batch_size=1)
```

It needs an initialized simplex/evaluated seed records. Treat its suggestions as
sequential; extra workers may still be useful for an initial design or other workflow
stages, but do not request parallel Nelder-Mead suggestions without support from the
installed version.

## Optimas

```python
from optimas.generators import AxSingleFidelityGenerator

generator = AxSingleFidelityGenerator(vocs=vocs)
gen_specs = GenSpecs(generator=generator, vocs=vocs, batch_size=4)
```

Common choices:

- `GridSamplingGenerator(vocs=vocs, n_steps=[...])` for a Cartesian grid. The `n_steps`
  order must match VOCS variable order.
- `AxSingleFidelityGenerator(vocs=vocs)` for standard Ax optimization.
- `AxMultiFidelityGenerator(vocs=vocs)` when VOCS includes the expected fidelity variable.
- `AxMultitaskGenerator` with explicit high-/low-fidelity `Task` configurations for a
  categorical task variable.

Confirm constructors against the installed Optimas version. Representative workflows use
fixed batches. Multitask restart can pass prior History as `H0`, provided it contains all
required fields.

## gpCAM

```python
from libensemble.gen_classes.gpCAM import GP_CAM

generator = GP_CAM(vocs, ask_max_iter=10, random_seed=1)
gen_specs = GenSpecs(generator=generator, vocs=vocs, batch_size=4)
```

Use `GP_CAM` for total-correlation acquisition or `GP_CAM_Covar` for posterior-covariance
sampling. Both generate an initial random batch and then train/update a GP. They are
batch-oriented; model fitting may dominate cheap simulations. Install a compatible
`gpcam` package before importing the module. Verify objective mapping and do not assume
constraint support.

## Selection safeguards

- Do not import Xopt's VOCS; use `gest_api.vocs.VOCS`.
- Do not replace a user's generator based solely on its name or apparent sampling behavior.
- Do not assume optional packages, GPU support, or optimizer backends exist.
- Do not claim support for constraints, categorical variables, multiple objectives,
  asynchronous updates, or restart unless verified for the concrete generator version.
- Initial evaluated records must contain variables plus all outputs consumed by the
  generator, not merely candidate coordinates.
