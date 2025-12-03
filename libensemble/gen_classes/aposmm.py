import copy
import warnings
from math import gamma, pi, sqrt
from typing import List

import numpy as np
from gest_api.vocs import VOCS
from numpy import typing as npt

from libensemble.generators import PersistentGenInterfacer
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP


class APOSMM(PersistentGenInterfacer):
    """
    APOSMM coordinates multiple local optimization runs, dramatically reducing time for
    discovering multiple minima on parallel systems.

    This *generator* adheres to the `Generator Standard <https://github.com/campa-consortium/generator_standard>`_.

    .. seealso::

        `https://doi.org/10.1007/s12532-017-0131-4 <https://doi.org/10.1007/s12532-017-0131-4>`_

    VOCS variables must include both regular and *_on_cube versions. E.g.,:

    ```python
    vars_std = {
        "var1": [-10.0, 10.0],
        "var2": [0.0, 100.0],
        "var3": [1.0, 50.0],
        "var1_on_cube": [0, 1.0],
        "var2_on_cube": [0, 1.0],
        "var3_on_cube": [0, 1.0],
    }
    variables_mapping = {
        "x": ["var1", "var2", "var3"],
        "x_on_cube": ["var1_on_cube", "var2_on_cube", "var3_on_cube"],
    }
    gen = APOSMM(vocs, 3, 3, variables_mapping=variables_mapping, ...)
    ```

    Getting started
    ---------------

    APOSMM requires a minimal sample before starting optimization. A random sample across the domain
    can either be retrieved via a `suggest()` call right after initialization, or the user can ingest
    a set of sample points via `ingest()`. The minimal sample size is specified via the `initial_sample_size`
    parameter. This many evaluated sample points *must* be provided to APOSMM before it will provide any
    local optimization points.

        ```python
        # Approach 1: Retrieve sample points via suggest()
        gen = APOSMM(vocs, max_active_runs=2, initial_sample_size=10)

        # ask APOSMM for some sample points
        initial_sample = gen.suggest(10)
        for point in initial_sample:
            point["f"] = func(point["x"])
        gen.ingest(initial_sample)

        # APOSMM will now provide local-optimization points.
        points = gen.suggest(10)

        # ----------------

        # Approach 2: Ingest pre-computed sample points via ingest()
        gen = APOSMM(vocs, max_active_runs=2, initial_sample_size=10)

        initial_sample = create_initial_sample()
        for point in initial_sample:
            point["f"] = func(point["x"])

        # provide APOSMM with sample points
        gen.ingest(initial_sample)

        # APOSMM will now provide local-optimization points.
        points = gen.suggest(10)

        ...
        ```

    *Important Note*: After the initial sample phase, APOSMM cannot accept additional "arbitrary"
    sample points that are not associated with local optimization runs.

        ```python
        gen = APOSMM(vocs, max_active_runs=2, initial_sample_size=10)

        # ask APOSMM for some sample points
        initial_sample = gen.suggest(10)
        for point in initial_sample:
            point["f"] = func(point["x"])
        gen.ingest(initial_sample)

        # APOSMM will now provide local-optimization points.
        points_from_aposmm = gen.suggest(10)
        for point in points_from_aposmm:
            point["f"] = func(point["x"])
        gen.ingest(points_from_aposmm)

        gen.ingest(another_sample)  # THIS CRASHES
        ```

    Parameters
    ----------
    vocs: VOCS
        The VOCS object, adhering to the VOCS interface from the Generator Standard.

    max_active_runs: int
        Bound on number of runs APOSMM is *concurrently* advancing.

    initial_sample_size: int

        Minimal sample points required before starting optimization.

        If `suggest(N)` is called first, APOSMM produces this many random sample points across the domain,
        with N <= initial_sample_size.

        If `ingest(sample)` is called first, multiple calls like `ingest(sample)` are required until
        the total number of points ingested is >= initial_sample_size.

        ```python
        gen = APOSMM(vocs, max_active_runs=2, initial_sample_size=10)

        # ask APOSMM for some sample points
        initial_sample = gen.suggest(10)
        for point in initial_sample:
            point["f"] = func(point["x"])
        gen.ingest(initial_sample)

        # APOSMM will now provide local-optimization points.
        points = gen.suggest(10)
        ...
        ```

    History: npt.NDArray = []
        An optional history of previously evaluated points.

    sample_points: npt.NDArray = None
        Included for compatibility with the underlying algorithm.
        Points to be sampled (original domain).
        If more sample points are needed by APOSMM during the course of the
        optimization, points will be drawn uniformly over the domain.

    localopt_method: str = "LN_BOBYQA"
        The local optimization method to use.

    rk_const: float = None
        Multiplier in front of the ``r_k`` value.
        If not provided, it will be set to ``0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi)``

    xtol_abs: float = 1e-6
        Localopt method's convergence tolerance.

    ftol_abs: float = 1e-6
        Localopt method's convergence tolerance.

    dist_to_bound_multiple: float = 0.5
        What fraction of the distance to the nearest boundary should the initial
        step size be in localopt runs.

    random_seed: int = 1
        Seed for the random number generator.
    """

    def _validate_vocs(self, vocs: VOCS):
        if len(vocs.constraints):
            warnings.warn("APOSMM does not support constraints in VOCS. Ignoring.")
        if len(vocs.constants):
            warnings.warn("APOSMM does not support constants in VOCS. Ignoring.")

    def __init__(
        self,
        vocs: VOCS,
        max_active_runs: int,
        initial_sample_size: int,
        History: npt.NDArray = [],
        sample_points: npt.NDArray = None,
        localopt_method: str = "LN_BOBYQA",
        rk_const: float = None,
        xtol_abs: float = 1e-6,
        ftol_abs: float = 1e-6,
        dist_to_bound_multiple: float = 0.5,
        random_seed: int = 1,
        **kwargs,
    ) -> None:

        from libensemble.gen_funcs.persistent_aposmm import aposmm

        self.VOCS = vocs

        gen_specs = {}
        gen_specs["user"] = {}
        persis_info = {}
        libE_info = {}
        gen_specs["gen_f"] = aposmm
        n = len(list(vocs.variables.keys()))

        if not rk_const:
            rk_const = 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi)

        FIELDS = [
            "initial_sample_size",
            "sample_points",
            "localopt_method",
            "rk_const",
            "xtol_abs",
            "ftol_abs",
            "dist_to_bound_multiple",
            "max_active_runs",
            "random_seed",
        ]

        for k in FIELDS:
            val = locals().get(k)
            if val is not None:
                gen_specs["user"][k] = val

        super().__init__(vocs, History, persis_info, gen_specs, libE_info, **kwargs)

        # Set bounds using the correct x mapping
        x_mapping = self.variables_mapping["x"]
        self.gen_specs["user"]["lb"] = np.array([vocs.variables[var].domain[0] for var in x_mapping])
        self.gen_specs["user"]["ub"] = np.array([vocs.variables[var].domain[1] for var in x_mapping])

        x_size = len(self.variables_mapping.get("x", []))
        x_on_cube_size = len(self.variables_mapping.get("x_on_cube", []))

        try:
            assert x_size > 0 and x_on_cube_size > 0
        except AssertionError:
            raise ValueError(
                """ User must provide a variables_mapping dictionary in the following format:

                    variables = {"core": [-3, 3], "edge": [-2, 2], "core_on_cube": [0, 1], "edge_on_cube": [0, 1]}
                    objectives = {"energy": "MINIMIZE"}

                    variables_mapping = {
                        "x": ["core", "edge"],
                        "x_on_cube": ["core_on_cube", "edge_on_cube"],
                        "f": ["energy"],
                    }
                """
            )
        try:
            assert x_size == x_on_cube_size
        except AssertionError:
            raise ValueError(
                "Within the variables_mapping dictionary, x and x_on_cube "
                + f"must have same length but got {x_size} and {x_on_cube_size}"
            )

        gen_specs["out"] = [
            ("x", float, x_size),
            ("x_on_cube", float, x_on_cube_size),
            ("sim_id", int),
            ("local_min", bool),
            ("local_pt", bool),
        ]

        gen_specs["persis_in"] = ["sim_id", "x", "x_on_cube", "f", "sim_ended"]
        if "components" in kwargs or "components" in gen_specs.get("user", {}):
            gen_specs["persis_in"].append("fvec")

        # SH - Need to know if this is gen_on_manager or not.
        self.persis_info["nworkers"] = gen_specs["user"].get("max_active_runs")
        self.all_local_minima = []
        self._suggest_idx = 0
        self._last_suggest = None
        self._ingest_buf = None
        self._n_buffd_results = 0
        self._told_initial_sample = False
        self._first_called_method = None
        self._last_call = None
        self._last_num_points = 0

    def _slot_in_data(self, results):
        """Slot in libE_calc_in and trial data into corresponding array fields. *Initial sample only!!*"""
        self._ingest_buf[self._n_buffd_results : self._n_buffd_results + len(results)] = results

    def _enough_initial_sample(self):
        return (
            self._n_buffd_results >= int(self.gen_specs["user"]["initial_sample_size"])
        ) or self._told_initial_sample

    def _ready_to_suggest_genf(self):
        """
        We're presumably ready to be suggested IF:
        - When we're working on the initial sample:
            - We have no _last_suggest cached
            - all points given out have returned AND we've been suggested *at least* as many points as we cached
        - When we're done with the initial sample:
            - we've been suggested *at least* as many points as we cached
            - we've just ingested some results
        """
        if not self._told_initial_sample and self._last_suggest is not None:
            cond = all([i in self._ingest_buf["sim_id"] for i in self._last_suggest["sim_id"]])
        else:
            cond = True
        return self._last_suggest is None or (cond and (self._suggest_idx >= len(self._last_suggest)))

    def suggest_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""

        if self._first_called_method is None:
            self._first_called_method = "suggest"
            self.gen_specs["user"]["generate_sample_points"] = True

        if self._ready_to_suggest_genf():
            self._suggest_idx = 0
            if self._last_call == "suggest" and num_points == 0 and self._last_num_points == 0:
                self.finalize()
                raise RuntimeError("Cannot suggest points since APOSMM is currently expecting to receive a sample")
            self._last_suggest = super().suggest_numpy(num_points)

            if self._last_suggest["local_min"].any():  # filter out local minima rows
                min_idxs = self._last_suggest["local_min"]
                self.all_local_minima.append(self._last_suggest[min_idxs])
                self._last_suggest = self._last_suggest[~min_idxs]

        if num_points > 0:  # we've been suggested for a selection of the last suggest
            results = np.copy(self._last_suggest[self._suggest_idx : self._suggest_idx + num_points])
            self._suggest_idx += num_points

        else:
            results = np.copy(self._last_suggest)
            self._last_suggest = None

        self._last_call = "suggest"
        self._last_num_points = num_points
        return results

    def ingest_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:

        if self._first_called_method is None:
            self._first_called_method = "ingest"
            self.gen_specs["user"]["generate_sample_points"] = False

        if (results is None and tag == PERSIS_STOP) or self._told_initial_sample:
            super().ingest_numpy(results, tag)
            self._last_call = "ingest"
            return

        # Initial sample buffering here:

        if self._n_buffd_results == 0:
            self._ingest_buf = np.zeros(self.gen_specs["user"]["initial_sample_size"], dtype=results.dtype)

        if not self._enough_initial_sample():
            self._slot_in_data(np.copy(results))
            self._n_buffd_results += len(results)

        if self._enough_initial_sample():
            if "sim_id" in results.dtype.names and not self._told_initial_sample:
                self._ingest_buf["sim_id"] = range(len(self._ingest_buf))
            super().ingest_numpy(self._ingest_buf, tag)
            self._told_initial_sample = True
            self._n_buffd_results = 0

        self._last_call = "ingest"

    def suggest_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima
