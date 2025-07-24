"""Generator classes providing points using sampling"""

import numpy as np
from generator_standard import Generator
from generator_standard.vocs import VOCS

from libensemble.generators import LibensembleGenerator

__all__ = [
    "UniformSample",
    "StandardSample",
]


class UniformSample(LibensembleGenerator):
    """
    This generator returns ``gen_specs["initial_batch_size"]`` uniformly
    sampled points the first time it is called. Afterwards, it returns the
    number of points given. This can be used in either a batch or asynchronous
    mode by adjusting the allocation function.
    """

    def __init__(self, VOCS: VOCS):
        super().__init__(VOCS)
        self.rng = np.random.default_rng(1)
        self.np_dtype = [(i, float) for i in self.VOCS.variables.keys()]

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.np_dtype)
        for trial in range(n_trials):
            for field in self.VOCS.variables.keys():
                out[trial][field] = self.rng.uniform(
                    self.VOCS.variables[field].domain[0], self.VOCS.variables[field].domain[1]
                )
        return out

    def ingest_numpy(self, calc_in):
        pass  # random sample so nothing to tell


class StandardSample(Generator):
    """
    This sampler only adheres to the complete standard interface, with no additional numpy methods.
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
