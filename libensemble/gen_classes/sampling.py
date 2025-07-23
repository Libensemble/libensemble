"""Generator classes providing points using sampling"""

import numpy as np
from generator_standard import Generator
from generator_standard.vocs import VOCS

from libensemble.generators import LibensembleGenerator

__all__ = [
    "UniformSample",
    "StandardSample",
]


class SampleBase(LibensembleGenerator):
    """Base class for sampling generators"""

    def _get_user_params(self, user_specs):
        """Extract user params"""
        self.ub = user_specs["ub"]
        self.lb = user_specs["lb"]
        self.n = len(self.lb)  # dimension
        assert isinstance(self.n, int), "Dimension must be an integer"
        assert isinstance(self.lb, np.ndarray), "lb must be a numpy array"
        assert isinstance(self.ub, np.ndarray), "ub must be a numpy array"


class UniformSample(SampleBase):
    """
    This generator returns ``gen_specs["initial_batch_size"]`` uniformly
    sampled points the first time it is called. Afterwards, it returns the
    number of points given. This can be used in either a batch or asynchronous
    mode by adjusting the allocation function.
    """

    def __init__(self, VOCS: VOCS, H=[], persis_info={}, gen_specs={}, libE_info=None, **kwargs):
        super().__init__(VOCS, H, persis_info, gen_specs, libE_info, **kwargs)
        self._get_user_params(gen_specs["user"])

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.gen_specs["out"])
        out["x"] = self.persis_info["rand_stream"].uniform(self.lb, self.ub, (n_trials, self.n))
        return out

    def ingest_numpy(self, calc_in):
        pass  # random sample so nothing to tell


class StandardSample(Generator):
    """
    This sampler only adheres to the complete standard interface, with no additional numpy methods.
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
