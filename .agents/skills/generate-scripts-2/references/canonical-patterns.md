# Canonical modern patterns

Use these complete patterns without repository access. Adapt names and values; do not copy
sample values when the user supplied different requirements. With built-in adapter
samplers, avoid a scalar VOCS variable named `x` when other variables are present: internal
vector field `x` is automatically mapped to all variables and can collide with it. Rename
that field (for example `x0`) and disclose the change; do not substitute an untested custom
generator just to preserve the ambiguous name.

## Built-in sampling and Python simulator

Use the runnable `examples/local_sampling.py` as the canonical full script. It uses
`LatinHypercubeSample` for one complete initial design, a pure-Python simulator, finite
result filtering, and manager-only output saving. Adapt variable names, bounds, objective,
worker count, and simulation budget to the user's requirements. Do not copy its sample
values as if they were user requirements.

Each `LatinHypercubeSample.suggest()` call forms a separate Latin hypercube. Use one initial
batch equal to the full design size when global stratification across the whole sample
matters. `initial_batch_size` controls the first request; `batch_size` controls later
requests. Sampling ignores feedback, so either synchronous or asynchronous delivery is
acceptable.

## Preloaded points

Use the runnable `examples/preloaded_points.py` as the canonical complete example.
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
        self.vocs = vocs
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

This is a generator fragment to combine with the complete calling-script pattern above.
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
Create a new module only when separation is needed. When rerunning in the same location,
use a new `ensemble_dir_path`, remove/archive the old workflow directory with user approval,
or intentionally set `reuse_output_dir=True`; a nonempty existing ensemble directory causes
startup failure by design. Resolve configurable paths at runtime
from CLI/config/environment or relative to `Path(__file__)`, then convert them to absolute
paths for registration/copying. Output filenames inside simulation directories may be
relative.
