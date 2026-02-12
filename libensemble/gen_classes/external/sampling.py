import numpy as np
from gest_api import Generator
from gest_api.vocs import VOCS

__all__ = [
    "UniformSample",
    "UniformSampleArray",
]


class UniformSample(Generator):
    """
    This sampler adheres to the gest-api VOCS interface and data structures (no numpy).

    Each variable is a scalar.
    """

    def __init__(self, vocs: VOCS):
        self.vocs = vocs
        self.rng = np.random.default_rng(1)
        super().__init__(vocs)

    def _validate_vocs(self, vocs):
        assert len(self.vocs.variable_names), "VOCS must contain variables."

    def suggest(self, n_trials):
        output = []
        for _ in range(n_trials):
            trial = {}
            for key in self.vocs.variables:
                trial[key] = self.rng.uniform(self.vocs.variables[key].domain[0], self.vocs.variables[key].domain[1])
            output.append(trial)
        return output

    def ingest(self, calc_in):
        pass  # random sample so nothing to tell


class UniformSampleArray(Generator):
    """
    This sampler adheres to the gest-api VOCS interface and data structures.

    Uses one array variable of any dimension. Array is a numpy array.
    """

    def __init__(self, vocs: VOCS):
        self.vocs = vocs
        self.rng = np.random.default_rng(1)
        super().__init__(vocs)

    def _validate_vocs(self, vocs):
        assert len(self.vocs.variables) == 1, "VOCS must contain exactly one variable."

    def suggest(self, n_trials):
        output = []
        key = list(self.vocs.variables.keys())[0]
        var = self.vocs.variables[key]
        for _ in range(n_trials):
            trial = {key: np.array([self.rng.uniform(bounds[0], bounds[1]) for bounds in var.domain])}
            output.append(trial)
        return output

    def ingest(self, calc_in):
        pass  # random sample so nothing to tell
