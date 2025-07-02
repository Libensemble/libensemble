"""Generator classes providing points using sampling"""

import numpy as np
from generator_standard.vocs import VOCS

from libensemble.generators import Generator

__all__ = [
    "UniformSample",
]


class UniformSample(Generator):
    """
    This sampler adheres to the complete standard.
    """

    def __init__(self, VOCS: VOCS, *args, **kwargs):
        self.VOCS = VOCS
        self.rng = np.random.default_rng(1)
        self._validate_vocs(VOCS)

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
