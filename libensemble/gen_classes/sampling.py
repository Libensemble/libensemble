"""Generator classes providing points using sampling"""

import numpy as np

from libensemble.generators import Generator, LibensembleGenerator

__all__ = [
    "UniformSample",
    "UniformSampleDicts",
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

    def __init__(self, _, persis_info, gen_specs, libE_info=None):
        self.persis_info = persis_info
        self.gen_specs = gen_specs
        self.libE_info = libE_info
        self._get_user_params(self.gen_specs["user"])

    def ask_numpy(self, n_trials):
        H_o = np.zeros(n_trials, dtype=self.gen_specs["out"])
        H_o["x"] = self.persis_info["rand_stream"].uniform(self.lb, self.ub, (n_trials, self.n))

        if "obj_component" in H_o.dtype.fields:  # needs H_o - needs to be created in here.
            H_o["obj_component"] = self.persis_info["rand_stream"].integers(
                low=0, high=self.gen_specs["user"]["num_components"], size=n_trials
            )
        return H_o

    def tell_numpy(self, calc_in):
        pass  # random sample so nothing to tell


# List of dictionaries format for ask (constructor currently using numpy still)
# Mostly standard generator interface for libE generators will use the ask/tell wrappers
# to the classes above. This is for testing a function written directly with that interface.
class UniformSampleDicts(Generator):
    """
    This generator returns ``gen_specs["initial_batch_size"]`` uniformly
    sampled points the first time it is called. Afterwards, it returns the
    number of points given. This can be used in either a batch or asynchronous
    mode by adjusting the allocation function.
    """

    def __init__(self, _, persis_info, gen_specs, libE_info=None):
        self.persis_info = persis_info
        self.gen_specs = gen_specs
        self.libE_info = libE_info
        self._get_user_params(self.gen_specs["user"])

    def ask(self, n_trials):
        H_o = []
        for _ in range(n_trials):
            # using same rand number stream
            trial = {"x": self.persis_info["rand_stream"].uniform(self.lb, self.ub, self.n)}
            H_o.append(trial)
        return H_o

    def tell(self, calc_in):
        pass  # random sample so nothing to tell

    # Duplicated for now
    def _get_user_params(self, user_specs):
        """Extract user params"""
        # b = user_specs["initial_batch_size"]
        self.ub = user_specs["ub"]
        self.lb = user_specs["lb"]
        self.n = len(self.lb)  # dimension
        assert isinstance(self.n, int), "Dimension must be an integer"
        assert isinstance(self.lb, np.ndarray), "lb must be a numpy array"
        assert isinstance(self.ub, np.ndarray), "ub must be a numpy array"
