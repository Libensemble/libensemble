from gest_api.vocs import VOCS
from gest_api import Generator
import numpy as np

__all__ = [
    "UniformSample",
    "UniformSampleArray",
]


class UniformSample(Generator):
    """
    This sampler adheres to the gest-api VOCS interface and data structures (no numpy).

    Each variable is a scalar.
    """

    def __init__(self, VOCS: VOCS):
        self.VOCS = VOCS
        self.rng = np.random.default_rng(1)
        super().__init__(VOCS)

    def _validate_vocs(self, VOCS):
        assert len(self.VOCS.variables), "VOCS must contain variables."

    def suggest(self, n_trials):
        output = []
        for _ in range(n_trials):
            trial = {}
            for key in self.VOCS.variables.keys():
                trial[key] = self.rng.uniform(self.VOCS.variables[key].domain[0], self.VOCS.variables[key].domain[1])
            output.append(trial)
        return output

    def ingest(self, calc_in):
        pass  # random sample so nothing to tell


class UniformSampleArray(Generator):
    """
    This sampler adheres to the gest-api VOCS interface and data structures.

    Uses one array variable of any dimension. Array is a numpy array.
    """

    def __init__(self, VOCS: VOCS):
        self.VOCS = VOCS
        self.rng = np.random.default_rng(1)
        super().__init__(VOCS)

    def _validate_vocs(self, VOCS):
        assert len(self.VOCS.variables) == 1, "VOCS must contain exactly one variable."

    def suggest(self, n_trials):
        output = []
        key = list(self.VOCS.variables.keys())[0]
        var = self.VOCS.variables[key]
        for _ in range(n_trials):
            trial = {key: np.array([
                self.rng.uniform(bounds[0], bounds[1])
                for bounds in var.domain
            ])}
            output.append(trial)
        return output

    def ingest(self, calc_in):
        pass  # random sample so nothing to tell
