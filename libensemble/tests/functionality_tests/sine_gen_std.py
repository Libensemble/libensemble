import numpy as np
from gest_api import Generator


class RandomSample(Generator):
    """
    This sampler accepts a gest-api VOCS object for configuration and returns random samples.
    """

    def __init__(self, vocs):
        self.variables = vocs.variables
        self.rng = np.random.default_rng(1)
        self._validate_vocs(vocs)

    def _validate_vocs(self, vocs):
        if not len(vocs.variables) > 0:
            raise ValueError("vocs must have at least one variable")

    def suggest(self, num_points):
        output = []
        for _ in range(num_points):
            trial = {}
            for key in self.variables.keys():
                trial[key] = self.rng.uniform(self.variables[key].domain[0], self.variables[key].domain[1])
            output.append(trial)
        return output
