"""Generator classes providing points using sampling"""

import numpy as np
from gest_api.vocs import VOCS

from libensemble.generators import LibensembleGenerator

__all__ = [
    "UniformSample",
]


class UniformSample(LibensembleGenerator):
    """
    Samples over the domain specified in the VOCS.

    Implements ``suggest()`` and ``ingest()`` identically to the other generators.
    """

    def __init__(self, VOCS: VOCS, random_seed: int = 1, *args, **kwargs):
        super().__init__(VOCS, *args, **kwargs)
        self.rng = np.random.default_rng(random_seed)

        self.n = len(list(self.VOCS.variables.keys()))
        self.np_dtype = [("x", float, (self.n))]
        self.lb = np.array([VOCS.variables[i].domain[0] for i in VOCS.variables])
        self.ub = np.array([VOCS.variables[i].domain[1] for i in VOCS.variables])

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.np_dtype)

        for i in range(n_trials):
            out[i]["x"] = self.rng.uniform(self.lb, self.ub, (self.n))

        return out

    def ingest_numpy(self, calc_in):
        pass  # random sample so nothing to tell
