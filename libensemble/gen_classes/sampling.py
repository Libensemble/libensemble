"""Generator classes providing points using sampling"""

import numpy as np
from gest_api.vocs import VOCS

from libensemble.generators import LibensembleGenerator

__all__ = [
    "UniformSample",
    "LatinHypercubeSample",
    "UniformSampleObjComponents",
    "UniformSampleWithVariableResources",
    "UniformSampleWithVarPrioritiesAndResources",
    "UniformSampleCancel",
]


class UniformSample(LibensembleGenerator):
    """
    Samples over the domain specified in the VOCS.

    Implements ``suggest()`` and ``ingest()`` identically to the other generators.
    """

    def __init__(self, vocs: VOCS, random_seed: int = 1, *args, **kwargs):
        super().__init__(vocs, *args, **kwargs)
        self.rng = np.random.default_rng(random_seed)

        self.n = len(list(self.vocs.variables.keys()))
        self.np_dtype = [(name, float) for name in self.vocs.variables.keys()]
        self.lb = np.array([vocs.variables[i].domain[0] for i in vocs.variables])
        self.ub = np.array([vocs.variables[i].domain[1] for i in vocs.variables])

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.np_dtype)

        for i in range(n_trials):
            vals = self.rng.uniform(self.lb, self.ub, (self.n))
            for j, name in enumerate(self.vocs.variables.keys()):
                out[i][name] = vals[j]

        return out

    def ingest_numpy(self, calc_in):
        pass  # random sample so nothing to tell


def _lhs_unit_cube(n, k, rng):
    """Generate ``k`` points in [0,1]^n using Latin hypercube sampling."""
    intervals = np.linspace(0, 1, k + 1)
    rand_source = rng.uniform(0, 1, (k, n))
    rand_pts = np.zeros((k, n))
    sample = np.zeros((k, n))

    a = intervals[:k]
    b = intervals[1:]
    for j in range(n):
        rand_pts[:, j] = rand_source[:, j] * (b - a) + a

    for j in range(n):
        sample[:, j] = rand_pts[rng.permutation(k), j]

    return sample


class LatinHypercubeSample(LibensembleGenerator):
    """
    Latin hypercube sample over the domain specified in the VOCS.

    All ``n_trials`` points are drawn at once from a single LHS design, so
    consecutive ``suggest()`` calls return new LHS designs (each independently
    space-filling, but not stratified together).
    """

    def __init__(self, vocs: VOCS, random_seed: int = 1, *args, **kwargs):
        super().__init__(vocs, *args, **kwargs)
        self.rng = np.random.default_rng(random_seed)

        self.n = len(list(self.vocs.variables.keys()))
        self.np_dtype = [(name, float) for name in self.vocs.variables.keys()]
        self.lb = np.array([vocs.variables[i].domain[0] for i in vocs.variables])
        self.ub = np.array([vocs.variables[i].domain[1] for i in vocs.variables])

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.np_dtype)

        sample = _lhs_unit_cube(self.n, n_trials, self.rng)
        scaled = sample * (self.ub - self.lb) + self.lb
        for j, name in enumerate(self.vocs.variables.keys()):
            out[name] = scaled[:, j]

        return out

    def ingest_numpy(self, calc_in):
        pass


class UniformSampleObjComponents(LibensembleGenerator):
    """
    Uniform random sample where each suggested point is replicated ``components``
    times so each objective component is evaluated separately. Each replicated row
    carries the same ``x`` plus an ``obj_component`` index, a shared ``pt_id``,
    and an independent random ``priority``.

    Used by component-aware solvers (e.g. POUNDERS, where each residual is its
    own evaluation). The ``obj_component``, ``pt_id``, and ``priority`` fields are
    libEnsemble H-array fields rather than VOCS objectives — downstream sim_f
    is expected to read ``obj_component`` and return the matching residual.
    """

    def __init__(self, vocs: VOCS, components: int, random_seed: int = 1, *args, **kwargs):
        super().__init__(vocs, *args, **kwargs)
        self.rng = np.random.default_rng(random_seed)
        self.components = components

        self.n = len(list(self.vocs.variables.keys()))
        self.np_dtype = [(name, float) for name in self.vocs.variables.keys()] + [
            ("priority", float),
            ("obj_component", int),
            ("pt_id", int),
        ]
        self.lb = np.array([vocs.variables[i].domain[0] for i in vocs.variables])
        self.ub = np.array([vocs.variables[i].domain[1] for i in vocs.variables])
        self._pt_id_offset = 0

    def suggest_numpy(self, n_trials):
        m = self.components
        out = np.zeros(n_trials * m, dtype=self.np_dtype)

        for i in range(n_trials):
            x = self.rng.uniform(self.lb, self.ub, (1, self.n))
            slc = slice(i * m, (i + 1) * m)
            for j, name in enumerate(self.vocs.variables.keys()):
                out[name][slc] = x[0, j]
            out["priority"][slc] = self.rng.uniform(0, 1, m)
            out["obj_component"][slc] = np.arange(m)
            out["pt_id"][slc] = self._pt_id_offset + i

        self._pt_id_offset += n_trials
        return out

    def ingest_numpy(self, calc_in):
        pass


class UniformSampleWithVariableResources(LibensembleGenerator):
    """
    Uniform random sample that also requests a random number of resource sets per
    evaluation (1 to ``max_resource_sets``). For testing/demonstrating variable
    resource allocation.

    .. note::
       ``resource_sets`` is a libEnsemble manager-side H-array field, not a
       VOCS variable. Whether the downstream libE manager honors it via this
       new generator-class path depends on alloc_specs; the classic gen_funcs
       path was tested with the default alloc.
    """

    def __init__(
        self, vocs: VOCS, max_resource_sets: int, random_seed: int = 1, *args, **kwargs
    ):
        super().__init__(vocs, *args, **kwargs)
        self.rng = np.random.default_rng(random_seed)
        self.max_rsets = max_resource_sets

        self.n = len(list(self.vocs.variables.keys()))
        self.np_dtype = [(name, float) for name in self.vocs.variables.keys()] + [
            ("resource_sets", int),
        ]
        self.lb = np.array([vocs.variables[i].domain[0] for i in vocs.variables])
        self.ub = np.array([vocs.variables[i].domain[1] for i in vocs.variables])

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.np_dtype)

        vals = self.rng.uniform(self.lb, self.ub, (n_trials, self.n))
        for j, name in enumerate(self.vocs.variables.keys()):
            out[name] = vals[:, j]
        out["resource_sets"] = self.rng.integers(1, self.max_rsets + 1, n_trials)

        return out

    def ingest_numpy(self, calc_in):
        pass


class UniformSampleWithVarPrioritiesAndResources(LibensembleGenerator):
    """
    Uniform random sample that emits an initial batch of ``initial_batch_size``
    points (each with one resource set and uniform priority), then on subsequent
    calls emits one point at a time with a random number of resource sets (1 to
    ``max_resource_sets``) and priority scaled by that count.

    .. note::
       Same caveat as ``UniformSampleWithVariableResources`` re: ``resource_sets``
       and ``priority`` being libEnsemble H-array fields rather than VOCS items.
    """

    def __init__(
        self,
        vocs: VOCS,
        max_resource_sets: int,
        initial_batch_size: int,
        random_seed: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(vocs, *args, **kwargs)
        self.rng = np.random.default_rng(random_seed)
        self.max_rsets = max_resource_sets
        self.initial_batch_size = initial_batch_size
        self._initial_emitted = False

        self.n = len(list(self.vocs.variables.keys()))
        self.np_dtype = [(name, float) for name in self.vocs.variables.keys()] + [
            ("resource_sets", int),
            ("priority", float),
        ]
        self.lb = np.array([vocs.variables[i].domain[0] for i in vocs.variables])
        self.ub = np.array([vocs.variables[i].domain[1] for i in vocs.variables])

    def suggest_numpy(self, n_trials):
        if not self._initial_emitted:
            b = self.initial_batch_size
            out = np.zeros(b, dtype=self.np_dtype)
            for i in range(b):
                x = self.rng.uniform(self.lb, self.ub, (1, self.n))
                for j, name in enumerate(self.vocs.variables.keys()):
                    out[name][i] = x[0, j]
            out["resource_sets"] = 1
            out["priority"] = 1.0
            self._initial_emitted = True
            return out

        out = np.zeros(1, dtype=self.np_dtype)
        x = self.rng.uniform(self.lb, self.ub)
        for j, name in enumerate(self.vocs.variables.keys()):
            out[name][0] = x[j]
        out["resource_sets"][0] = self.rng.integers(1, self.max_rsets + 1)
        out["priority"][0] = 10 * out["resource_sets"][0]
        return out

    def ingest_numpy(self, calc_in):
        pass


class UniformSampleCancel(LibensembleGenerator):
    """
    Uniform random sample but every 10th point in each batch is emitted with
    ``cancel_requested=True``. For testing immediate-cancellation paths.

    .. note::
       ``cancel_requested`` is a libEnsemble H-array field, not a VOCS variable.
       Same caveat as the resource samplers.
    """

    def __init__(self, vocs: VOCS, random_seed: int = 1, *args, **kwargs):
        super().__init__(vocs, *args, **kwargs)
        self.rng = np.random.default_rng(random_seed)

        self.n = len(list(self.vocs.variables.keys()))
        self.np_dtype = [(name, float) for name in self.vocs.variables.keys()] + [
            ("cancel_requested", bool),
        ]
        self.lb = np.array([vocs.variables[i].domain[0] for i in vocs.variables])
        self.ub = np.array([vocs.variables[i].domain[1] for i in vocs.variables])

    def suggest_numpy(self, n_trials):
        out = np.zeros(n_trials, dtype=self.np_dtype)

        vals = self.rng.uniform(self.lb, self.ub, (n_trials, self.n))
        for j, name in enumerate(self.vocs.variables.keys()):
            out[name] = vals[:, j]
        for i in range(n_trials):
            if i % 10 == 0:
                out["cancel_requested"][i] = True

        return out

    def ingest_numpy(self, calc_in):
        pass
