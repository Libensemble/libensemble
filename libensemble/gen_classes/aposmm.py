import copy
import warnings
from math import gamma, pi, sqrt
from typing import List

import numpy as np
from gest_api.vocs import VOCS
from numpy import typing as npt

from libensemble.generators import LibensembleGenerator
from libensemble.utils.misc import unmap_numpy_array


class APOSMM(LibensembleGenerator):
    """
    APOSMM coordinates multiple local optimization runs, dramatically reducing time for
    discovering multiple minima on parallel systems.

    This *generator* adheres to the `Generator Standard <https://github.com/campa-consortium/generator_standard>`_.

    .. seealso::

        `https://doi.org/10.1007/s12532-017-0131-4 <https://doi.org/10.1007/s12532-017-0131-4>`_

    VOCS variables must include both regular and ``*_on_cube`` versions. E.g.,:

    .. code-block:: python

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

    Getting started
    ---------------

    APOSMM requires a minimal sample before starting optimization. A random sample across the domain
    can either be retrieved via a ``suggest()`` call right after initialization, or the user can ingest
    a set of sample points via ``ingest()``. The minimal sample size is specified via the ``initial_sample_size``
    parameter. This many evaluated sample points *must* be provided to APOSMM before it will provide any
    local optimization points.

    .. code-block:: python

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


    .. important::
        After the initial sample phase, APOSMM cannot accept additional "arbitrary"
        sample points that are not associated with local optimization runs.


    .. code-block:: python

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

    Parameters
    ----------
    vocs: ``VOCS``
        The VOCS object, adhering to the VOCS interface from the Generator Standard.

    max_active_runs: ``int``
        Bound on number of runs APOSMM is *concurrently* advancing.

    initial_sample_size: ``int``

        Minimal sample points required before starting optimization.

        If ``suggest(N)`` is called first, APOSMM produces this many random sample points across the domain,
        with ``N <= initial_sample_size``.

        If ``ingest(sample)`` is called first, multiple calls like ``ingest(sample)`` are required until
        the total number of points ingested is ``>= initial_sample_size``.

    History: ``npt.NDArray`` = ``[]``
        An optional history of previously evaluated points.

    sample_points: ``npt.NDArray`` = ``None``
        Included for compatibility with the underlying algorithm.
        Points to be sampled (original domain).
        If more sample points are needed by APOSMM during the course of the
        optimization, points will be drawn uniformly over the domain.

    localopt_method: ``str`` = "scipy_Nelder-Mead" (scipy) or "LN_BOBYQA" (nlopt)
        The local optimization method to use. Others being added over time.

    mu: ``float`` = ``1e-8``
        Distance from the boundary that all localopt starting points must satisfy

    nu: ``float`` = ``1e-8``
        Distance from identified minima that all starting points must satisfy

    rk_const: ``float`` = ``None``
        Multiplier in front of the ``r_k`` value.
        If not provided, it will be set to ``0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi)``

    xtol_abs: ``float`` = ``1e-6``
        Localopt method's convergence tolerance.

    ftol_abs: ``float`` = ``1e-6``
        Localopt method's convergence tolerance.

    opt_return_codes: ``list[int]`` = ``[0]``
        scipy only: List of return codes that determine if a point should be ruled a local minimum.

    dist_to_bound_multiple: ``float`` = ``0.5``
        What fraction of the distance to the nearest boundary should the initial
        step size be in localopt runs.

    random_seed: ``int`` = ``1``
        Seed for the random number generator.
    """

    returns_id = True

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
        localopt_method: str = "scipy_Nelder-Mead",
        rk_const: float = None,
        xtol_abs: float = 1e-6,
        ftol_abs: float = 1e-6,
        opt_return_codes: list[int] = [0],
        mu: float = 1e-8,
        nu: float = 1e-8,
        dist_to_bound_multiple: float = 0.5,
        random_seed: int = 1,
        **kwargs,
    ) -> None:

        from libensemble.gen_funcs.aposmm_localopt_support import LocalOptInterfacer
        from libensemble.gen_funcs.persistent_aposmm import (
            add_k_sample_points_to_local_H,
            add_to_local_H,
            decide_where_to_start_localopt,
            initialize_APOSMM,
            initialize_children,
            initialize_dists_and_inds,
            update_history_dist,
            update_history_optimal,
        )

        # Store references to the functions we'll call later
        self._add_k_sample_points = add_k_sample_points_to_local_H
        self._add_to_local_H = add_to_local_H
        self._decide_where_to_start = decide_where_to_start_localopt
        self._initialize_dists_and_inds = initialize_dists_and_inds
        self._update_history_dist = update_history_dist
        self._update_history_optimal = update_history_optimal
        self._LocalOptInterfacer = LocalOptInterfacer

        self.vocs = vocs

        gen_specs = {}
        gen_specs["user"] = {}
        persis_info = {}
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
            "mu",
            "nu",
            "opt_return_codes",
            "dist_to_bound_multiple",
            "max_active_runs",
            "random_seed",
        ]

        for k in FIELDS:
            val = locals().get(k)
            if val is not None:
                gen_specs["user"][k] = val

        super().__init__(vocs, History, persis_info, gen_specs, {}, **kwargs)

        # APOSMM manages sim_id internally — don't remap to _id
        self.variables_mapping.pop("sim_id", None)

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

        # Initialize APOSMM internal state directly (no subprocess)
        user_specs = gen_specs["user"]
        libE_info = {"comm": []}  # no comm needed in direct mode
        self._n, self._n_s, self._rk_const, self._ld, self._mu, self._nu, _, self.local_H = initialize_APOSMM(
            History, user_specs, libE_info
        )
        (
            self._local_opters,
            self._sim_id_to_child_inds,
            self._run_order,
            self._run_pts,
            self._total_runs,
            self._ended_runs,
            self._fields_to_pass,
        ) = initialize_children(user_specs)

        self._user_specs = user_specs
        self._max_active_runs = max_active_runs

        # Build reverse mapping: VOCS field name -> (internal_name, index)
        self._reverse_mapping = {}
        for internal_name, vocs_names in self.variables_mapping.items():
            for i, vocs_name in enumerate(vocs_names):
                self._reverse_mapping[vocs_name] = (internal_name, i, len(vocs_names))

        self.all_local_minima = []
        self._told_initial_sample = False
        self._first_called_method = None
        self._pending_results = None
        self._first_pass = True
        self._n_r = 0  # number of results received in last ingest
        self._initial_sample_generated = False
        self._initial_suggest_idx = 0  # tracks how many initial sample points have been handed out

    def _map_to_internal(self, results):
        """Map VOCS-named structured array to internal APOSMM field names (x, x_on_cube, f, sim_id)."""
        if results is None or len(results) == 0:
            return results
        # If already has internal names, return as-is
        if "x" in results.dtype.names and "f" in results.dtype.names:
            return results

        n_rows = len(results)
        # Build dtype for internal array
        internal_fields = []
        added = set()
        for vocs_name in results.dtype.names:
            if vocs_name in self._reverse_mapping:
                internal_name, _, size = self._reverse_mapping[vocs_name]
                if internal_name not in added:
                    if size > 1:
                        internal_fields.append((internal_name, float, size))
                    else:
                        internal_fields.append((internal_name, float))
                    added.add(internal_name)
            elif vocs_name == "_id":
                if "sim_id" not in added:
                    internal_fields.append(("sim_id", int))
                    added.add("sim_id")
            elif vocs_name == "sim_id":
                if "sim_id" not in added:
                    internal_fields.append(("sim_id", int))
                    added.add("sim_id")

        out = np.zeros(n_rows, dtype=internal_fields)
        has_sim_id = "sim_id" in results.dtype.names
        for vocs_name in results.dtype.names:
            if vocs_name in self._reverse_mapping:
                internal_name, idx, size = self._reverse_mapping[vocs_name]
                if size > 1:
                    out[internal_name][:, idx] = results[vocs_name]
                else:
                    out[internal_name] = results[vocs_name]
            elif vocs_name == "sim_id":
                out["sim_id"] = results["sim_id"]
            elif vocs_name == "_id" and not has_sim_id:
                out["sim_id"] = results["_id"]

        return out

    def _slot_in_data(self, results):
        """Slot ingested results into local_H during initial sample phase."""
        n_s_before = self._n_s
        n_new = len(results)
        old_len = len(self.local_H)
        needed = n_s_before + n_new
        if needed > old_len:
            self.local_H.resize(needed, refcheck=False)
            self._initialize_dists_and_inds(self.local_H, needed - old_len)

        for i, row in enumerate(results):
            idx = n_s_before + i
            self.local_H["sim_id"][idx] = idx
            for name in results.dtype.names:
                if name == "sim_id":
                    continue
                if name in self.local_H.dtype.names:
                    self.local_H[name][idx] = row[name]
            self.local_H["sim_ended"][idx] = True
        self._n_s += n_new
        self._update_history_dist(self.local_H, self._n)

    def suggest_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        out_fields = [i[0] for i in self.gen_specs["out"]]

        if self._first_called_method is None:
            self._first_called_method = "suggest"

        # Initial sample phase: generate random points once, return in batches
        if not self._told_initial_sample:
            if not self._initial_sample_generated:
                total = self._user_specs["initial_sample_size"]
                self._add_k_sample_points(
                    total, self._user_specs, self.persis_info,
                    self._n, [], self.local_H, self._sim_id_to_child_inds,
                )
                self._initial_sample_generated = True
                self._initial_suggest_idx = 0

            k = num_points if num_points > 0 else (self._user_specs["initial_sample_size"] - self._initial_suggest_idx)
            start = self._initial_suggest_idx
            end = min(start + k, self._user_specs["initial_sample_size"])
            result = self.local_H[start:end][out_fields].copy()
            self._initial_suggest_idx = end
            return unmap_numpy_array(result, self.variables_mapping)

        # Main optimization phase
        new_opt_inds = []
        new_inds = []

        # Process any pending ingested results through local optimizers
        if self._pending_results is not None:
            from libensemble.gen_funcs.aposmm_localopt_support import ConvergedMsg

            calc_in = self._pending_results
            self._pending_results = None

            # Update local_H with received results
            for row in calc_in:
                sim_id = int(row["sim_id"])
                self.local_H[sim_id]["sim_ended"] = True
                for name in calc_in.dtype.names:
                    if name in self.local_H.dtype.names:
                        self.local_H[name][sim_id] = row[name]
            self._n_s = int(np.sum(~self.local_H["local_pt"][:len(self.local_H)]))
            self._update_history_dist(self.local_H, self._n)

            for row in calc_in:
                sim_id = int(row["sim_id"])
                if self._sim_id_to_child_inds.get(sim_id):
                    for child_idx in self._sim_id_to_child_inds[sim_id]:
                        if child_idx not in self._local_opters:
                            continue
                        x_new = self._local_opters[child_idx].iterate(row[self._fields_to_pass])
                        if isinstance(x_new, ConvergedMsg):
                            x_opt = x_new.x
                            opt_flag = x_new.opt_flag
                            opt_ind = self._update_history_optimal(
                                x_opt, opt_flag, self.local_H, self._run_order[child_idx],
                            )
                            new_opt_inds.append(opt_ind)
                            self._local_opters.pop(child_idx)
                            self._ended_runs.append(child_idx)
                        else:
                            self._add_to_local_H(self.local_H, x_new, self._user_specs, local_flag=1, on_cube=True)
                            new_inds.append(len(self.local_H) - 1)
                            self._run_order[child_idx].append(self.local_H[-1]["sim_id"])
                            self._run_pts[child_idx].append(x_new)
                            sid = self.local_H[-1]["sim_id"]
                            if sid in self._sim_id_to_child_inds:
                                self._sim_id_to_child_inds[sid] += (child_idx,)
                            else:
                                self._sim_id_to_child_inds[sid] = (child_idx,)

        # Decide where to start new local optimization runs
        starting_inds = self._decide_where_to_start(
            self.local_H, self._n, self._n_s, self._rk_const, self._ld, self._mu, self._nu,
        )

        for ind in starting_inds:
            if len([p for p in self._local_opters.values() if p.is_running]) < self._max_active_runs:
                self.local_H["started_run"][ind] = 1
                local_opter = self._LocalOptInterfacer(
                    self._user_specs,
                    self.local_H[ind]["x_on_cube"],
                    self.local_H[ind]["f"] if "f" in self._fields_to_pass else self.local_H[ind]["fvec"],
                    self.local_H[ind]["grad"] if "grad" in self._fields_to_pass else None,
                )
                self._local_opters[self._total_runs] = local_opter
                x_new = local_opter.iterate(self.local_H[ind][self._fields_to_pass])
                self._add_to_local_H(self.local_H, x_new, self._user_specs, local_flag=1, on_cube=True)
                new_inds.append(len(self.local_H) - 1)
                self._run_order[self._total_runs] = [ind, self.local_H[-1]["sim_id"]]
                self._run_pts[self._total_runs] = [self.local_H["x_on_cube"], x_new]
                sid = self.local_H[-1]["sim_id"]
                if sid in self._sim_id_to_child_inds:
                    self._sim_id_to_child_inds[sid] += (self._total_runs,)
                else:
                    self._sim_id_to_child_inds[sid] = (self._total_runs,)
                self._total_runs += 1

        # Fill remaining slots with sample points
        if self._first_pass:
            num_samples = self._max_active_runs - 1 - len(new_inds)
            self._first_pass = False
        else:
            num_samples = self._n_r - len(new_inds)

        if num_samples > 0:
            self._add_k_sample_points(
                num_samples, self._user_specs, self.persis_info,
                self._n, [], self.local_H, self._sim_id_to_child_inds,
            )
            new_inds = new_inds + list(range(len(self.local_H) - num_samples, len(self.local_H)))

        all_inds = new_inds + new_opt_inds
        if len(all_inds) == 0:
            return np.zeros(0, dtype=[(name, self.local_H.dtype[name]) for name in out_fields])

        result = self.local_H[all_inds][out_fields].copy()

        # Track local minima for suggest_updates()
        if result["local_min"].any():
            min_idxs = result["local_min"]
            self.all_local_minima.append(result[min_idxs].copy())

        return unmap_numpy_array(result, self.variables_mapping)

    def ingest_numpy(self, results: npt.NDArray, tag: int = 0) -> None:
        """Send the results of evaluations to the generator."""

        if results is None:
            return

        if self._first_called_method is None:
            self._first_called_method = "ingest"

        results = self._map_to_internal(results)

        if not self._told_initial_sample:
            # Initial sample phase: slot data into local_H
            self._slot_in_data(results)
            if self._n_s >= self._user_specs["initial_sample_size"]:
                self._told_initial_sample = True
            return

        # Main phase: buffer results for processing in next suggest call
        self._n_r = len(results)
        self._pending_results = results.copy()

    def suggest_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima

    def finalize(self) -> None:
        """Stop all local optimizer processes."""
        for _, p in self._local_opters.items():
            p.destroy()
        self._local_opters.clear()
