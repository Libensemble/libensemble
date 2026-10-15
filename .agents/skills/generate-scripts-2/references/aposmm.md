# Standardized APOSMM

APOSMM runs multiple local optimizations concurrently to discover multiple minima. It is
not the default choice for a single global optimum, constrained optimization, categorical
variables, or multi-objective optimization.

## SciPy template

```python
import numpy as np

from gest_api.vocs import VOCS
from libensemble import Ensemble
from libensemble.specs import GenSpecs, LibeSpecs, SimSpecs


def six_hump(inputs: dict, **kwargs) -> dict:
    x0, x1 = inputs["x0"], inputs["x1"]
    return {
        "f": (4 - 2.1 * x0**2 + x0**4 / 3) * x0**2
        + x0 * x1
        + (-4 + 4 * x1**2) * x1**2
    }


if __name__ == "__main__":
    import libensemble.gen_funcs

    libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
    from libensemble.gen_classes import APOSMM

    vocs = VOCS(
        variables={"x0": [-3.0, 3.0], "x1": [-2.0, 2.0]},
        objectives={"f": "MINIMIZE"},
    )
    generator = APOSMM(
        vocs,
        max_active_runs=4,
        initial_sample_size=40,
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        variables_mapping={"x": ["x0", "x1"], "f": ["f"]},
        random_seed=1,
    )
    ensemble = Ensemble(
        sim_specs=SimSpecs(simulator=six_hump, vocs=vocs),
        gen_specs=GenSpecs(
            generator=generator,
            vocs=vocs,
            initial_batch_size=40,
            batch_size=4,
        ),
        libE_specs=LibeSpecs(comms="local", nworkers=4, safe_mode=True),
    )
    H, _, _ = ensemble.run(sim_max=500)

    if ensemble.is_manager:
        completed_minima = H[
            H["sim_ended"] & H["local_min"] & np.isfinite(H["f"])
        ]
        ensemble.save_output("aposmm_results", append_attrs=False)
        print(completed_minima[["x0", "x1", "f"]])
```

The standardized APOSMM class currently requires one compatibility setup through the
legacy backend registry: set `libensemble.gen_funcs.rc.aposmm_optimizers` before importing
`APOSMM`. Keep this isolated in the main block; the generated workflow still uses the
standardized generator API. Use `"nlopt"` for an NLopt method and ensure the selected
backend package is installed.

## Required design choices

- `max_active_runs`: concurrent local optimizer runs; size it to useful simulation
  concurrency rather than blindly copying worker count.
- `initial_sample_size`: evaluated points APOSMM waits for before local optimization.
- `localopt_method`: must match the configured backend and objective information.
- `variables_mapping`: map APOSMM's vector `x` to ordered VOCS variable names and scalar
  `f` to the objective. Add mappings required by gradient/residual methods.
- `initial_batch_size`: normally equal to `initial_sample_size` when APOSMM generates the
  initial design itself.

APOSMM can obtain initial data from its own sample, `sample_points`, warm-start History,
or pre-ingested records. Do not feed arbitrary new sample points after local optimization
has begun.

## Constraints and constants

Current standardized APOSMM ignores VOCS constraints and constants and warns about them.
Do not present it as a constrained optimizer. Ask the user to select a compatible method
or transform the problem only with explicit approval.

## Local optimizer guidance

- `scipy_Nelder-Mead`: derivative-free baseline; SciPy required.
- SciPy gradient methods require the corresponding derivative output/mapping.
- NLopt methods require `nlopt` and method-specific stopping tolerances/return codes.
- PETSc/TAO, DFO-LS, IBCDFO, and external local optimizers require their own packages and
  output contracts.

Do not guess return codes, gradient fields, residual shapes, or tolerances. Confirm them
for the chosen backend/version.

## Results

`local_min` marks minima identified by APOSMM. Filter by both `sim_ended` and `local_min`,
then require finite objectives. Generated-but-unevaluated rows are not minima to report.
