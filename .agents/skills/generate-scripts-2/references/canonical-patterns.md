# Canonical modern patterns

Use these complete patterns without repository access. Adapt names and values; do not copy
sample values when the user supplied different requirements.

## Built-in sampling and Python simulator

```python
import numpy as np
from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes.sampling import UniformSample
from libensemble.specs import GenSpecs, LibeSpecs, SimSpecs


def simulate(inputs: dict, **kwargs) -> dict:
    x0 = inputs["x0"]
    x1 = inputs["x1"]
    return {"f": x0**2 + x1**2}


if __name__ == "__main__":
    vocs = VOCS(
        variables={"x0": [-3.0, 3.0], "x1": [-2.0, 2.0]},
        objectives={"f": "MINIMIZE"},
    )

    ensemble = Ensemble(
        sim_specs=SimSpecs(simulator=simulate, vocs=vocs),
        gen_specs=GenSpecs(
            generator=UniformSample(vocs, random_seed=1),
            vocs=vocs,
            initial_batch_size=4,
            batch_size=4,
        ),
        libE_specs=LibeSpecs(comms="local", nworkers=4, safe_mode=True),
    )

    H, _, flag = ensemble.run(sim_max=100)

    if ensemble.is_manager:
        completed = H[H["sim_ended"]]
        finite = completed[np.isfinite(completed["f"])]
        ensemble.save_output("sampling_results", append_attrs=False)
        if len(finite):
            best = finite[np.argmin(finite["f"])]
            print(f"Best: x0={best['x0']}, x1={best['x1']}, f={best['f']}")
        if flag != 0:
            raise RuntimeError(f"libEnsemble exited with flag {flag}; partial results saved")
```

Use `LatinHypercubeSample` from the same module for a space-filling batch. Each separate
LHS `suggest()` call forms a separate Latin hypercube; use one initial batch equal to the
full design size when global stratification matters.

`initial_batch_size` controls the first request; `batch_size` controls later requests.
Sampling ignores feedback, so either synchronous or asynchronous delivery is acceptable.

## Preloaded points

```python
from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import PreloadedSampleGenerator
from libensemble.specs import GenSpecs, LibeSpecs, SimSpecs


def simulate(inputs: dict, **kwargs) -> dict:
    return {"f": inputs["x0"] ** 2 + inputs["x1"] ** 2}


if __name__ == "__main__":
    points = [
        {"x0": -1.0, "x1": 0.5},
        {"x0": 0.0, "x1": 0.0},
        {"x0": 1.0, "x1": 0.5},
    ]
    vocs = VOCS(
        variables={"x0": [-3.0, 3.0], "x1": [-2.0, 2.0]},
        objectives={"f": "MINIMIZE"},
    )

    ensemble = Ensemble(
        sim_specs=SimSpecs(simulator=simulate, vocs=vocs),
        gen_specs=GenSpecs(
            generator=PreloadedSampleGenerator(points, vocs=vocs, batch_size=2),
            vocs=vocs,
        ),
        libE_specs=LibeSpecs(comms="local", nworkers=2, safe_mode=True),
    )
    H, _, flag = ensemble.run(sim_max=len(points))
    if ensemble.is_manager:
        ensemble.save_output("preloaded_results", append_attrs=False)
        if flag != 0:
            raise RuntimeError(f"libEnsemble exited with flag {flag}; partial results saved")
```

`PreloadedSampleGenerator` accepts a list of dictionaries or a NumPy structured array and
returns an empty suggestion after exhaustion. Use it instead of a pre-generated-work
allocator. Ensure every point has the simulator's required variable fields and any VOCS
constants; constants are not guaranteed to be injected by the generator.

## Custom standardized generator

```python
import numpy as np
from gest_api import Generator
from gest_api.vocs import VOCS


class RandomGenerator(Generator):
    def __init__(self, vocs: VOCS, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        super().__init__(vocs)

    def _validate_vocs(self, vocs: VOCS) -> None:
        if not vocs.variables:
            raise ValueError("VOCS must define variables")

    def suggest(self, n_trials: int) -> list[dict]:
        return [
            {
                **{
                    name: self.rng.uniform(variable.domain[0], variable.domain[1])
                    for name, variable in self.vocs.variables.items()
                },
                **self.vocs.constants,
            }
            for _ in range(n_trials)
        ]

    def ingest(self, calc_in: list[dict]) -> None:
        pass
```

A custom adaptive generator uses `ingest(calc_in)` to update state from completed records.
Do not use this example for discrete/categorical/vector variables without implementing
sampling for their actual VOCS variable classes.

## VOCS capabilities

```python
vocs = VOCS(
    variables={"x": [0.0, 1.0], "y": [-2.0, 2.0]},
    objectives={"cost": "MINIMIZE"},
    constraints={"temperature": ["LESS_THAN", 100.0]},
    observables=["runtime"],
    constants={"material": 2.0},
)
```

The simulator receives constants only when suggestions populate those fields; VOCS alone
may define History fields without assigning their values. Wrap or configure the generator
to add `vocs.constants`, and test them. The simulator must return `cost`, `temperature`, and
`runtime` when required. Confirm the selected generator supports the
specified variable types, multiple objectives, and constraints; declaring them in VOCS
does not make every algorithm support them.

## Script organization

Keep a small pure-Python simulator in the calling script. Put substantial parsing,
application control, or domain logic in an existing simulator module when one exists.
Create a new module only when separation is needed. Resolve configurable paths at runtime
from CLI/config/environment or relative to `Path(__file__)`, then convert them to absolute
paths for registration/copying. Output filenames inside simulation directories may be
relative.
