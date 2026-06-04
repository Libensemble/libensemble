"""
APOSMM generator class that directly implements the APOSMM algorithm
without wrapping the persistent generator function via QCommProcess.

Individual local optimization runs still use multiprocessing.Process
(via LocalOptInterfacer), but the outer APOSMM state machine is embedded
directly in the suggest_numpy/ingest_numpy methods.
"""

import copy
import warnings
from math import gamma, pi, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
from gest_api.vocs import VOCS
from numpy import typing as npt

from libensemble.gen_funcs.aposmm_localopt_support import ConvergedMsg, LocalOptInterfacer
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
from libensemble.generators import LibensembleGenerator
from libensemble.utils.misc import unmap_numpy_array


class APOSMM(LibensembleGenerator):
    """
    APOSMM coordinates multiple local optimization runs, dramatically reducing time for
    discovering multiple minima on parallel systems.

    This implementation directly embeds the APOSMM state machine in the ``suggest``/``ingest``
    methods, without wrapping the persistent generator function via subprocess communication.
    Individual local optimization runs still use subprocesses (via ``LocalOptInterfacer``).

    This *generator* adheres to the `Generator Standard <https://github.com/campa-consortium/generator_standard>`_.

    .. seealso::

        `https://doi.org/10.1007/s12532-017-0131-4 <https://doi.org/10.1007/s12532-017-0131-4>`_

    VOCS variables must include both regular and ``*_on_cube`` versions. E.g.,:

    .. code-block:: python

        vars_std = {
            "var1": [-10.0, 10.0],
            "var2": [0.0, 100.0],
            "var1_on_cube": [0, 1.0],
            "var2_on_cube": [0, 1.0],
        }
        variables_mapping = {
            "x": ["var1", "var2"],
            "x_on_cube": ["var1_on_cube", "var2_on_cube"],
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

    Parameters
    ----------
    vocs: ``VOCS``
        The VOCS object, adhering to the VOCS interface from the Generator Standard.

    max_active_runs: ``int``
        Bound on number of runs APOSMM is *concurrently* advancing.

    initial_sample_size: ``int``
        Minimal sample points required before starting optimization.

    History: ``npt.NDArray`` = ``[]``
        An optional history of previously evaluated points (H0).

    sample_points: ``npt.NDArray`` = ``None``
        Points to be sampled (original domain). If more sample points are needed
        by APOSMM during the course of the optimization, points will be drawn
        uniformly over the domain.

    localopt_method: ``str`` = ``"scipy_Nelder-Mead"``
        The local optimization method to use. Supported values:

        - NLopt: ``"LN_SBPLX"``, ``"LN_BOBYQA"``, ``"LN_COBYLA"``, ``"LN_NEWUOA"``,
          ``"LN_NELDERMEAD"``, ``"LD_MMA"``
        - PETSc/TAO: ``"pounders"``, ``"blmvm"``, ``"nm"``
        - SciPy: ``"scipy_Nelder-Mead"``, ``"scipy_COBYLA"``, ``"scipy_BFGS"``
        - DFO-LS: ``"dfols"``
        - IBCDFO: ``"ibcdfo_pounders"``, ``"ibcdfo_manifold_sampling"``
        - External: ``"external_localopt"``

    mu: ``float`` = ``1e-8``
        Distance from the boundary that all localopt starting points must satisfy.

    nu: ``float`` = ``1e-8``
        Distance from identified minima that all starting points must satisfy.

    rk_const: ``float`` = ``None``
        Multiplier in front of the ``r_k`` value. If not provided, defaults to
        ``0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi)``.

    xtol_abs: ``float`` = ``1e-6``
        Absolute x tolerance for NLopt local optimizer convergence.

    ftol_abs: ``float`` = ``1e-6``
        Absolute f tolerance for NLopt local optimizer convergence.

    xtol_rel: ``float`` = ``None``
        Relative x tolerance for NLopt local optimizer convergence.

    ftol_rel: ``float`` = ``None``
        Relative f tolerance for NLopt local optimizer convergence.

    grtol: ``float`` = ``None``
        Gradient tolerance for PETSc/TAO local optimizer convergence.

    gatol: ``float`` = ``None``
        Absolute gradient tolerance for PETSc/TAO local optimizer convergence.

    opt_return_codes: ``list[int]`` = ``[0]``
        SciPy only: List of return codes that determine if a point should be ruled
        a local minimum. E.g., Nelder-Mead and BFGS use ``[0]``, COBYLA uses ``[1]``.

    dist_to_bound_multiple: ``float`` = ``0.5``
        What fraction of the distance to the nearest boundary should the initial
        step size be in localopt runs.

    random_seed: ``int`` = ``1``
        Seed for the random number generator.

    stop_after_k_minima: ``int`` = ``None``
        Stop after finding this many local minima.

    stop_after_k_runs: ``int`` = ``None``
        Stop after this many local optimization runs have ended.

    periodic: ``bool`` = ``False``
        If ``True``, treat the domain as periodic. Points wrapping past the boundary
        are reflected back into the unit cube.

    run_max_eval: ``int`` = ``None``
        Maximum number of function evaluations per local optimization run.
        Defaults to ``1000 * n`` for NLopt/TAO and ``100 * (n + 1)`` for IBCDFO.

    lhs_divisions: ``int`` = ``0``
        Number of Latin hypercube sampling divisions for the sample points.
        0 or 1 results in uniform random sampling.

    scipy_kwargs: ``dict`` = ``None``
        Additional keyword arguments to pass to the SciPy local optimizer.

    dfols_kwargs: ``dict`` = ``None``
        Additional keyword arguments to pass to the DFO-LS local optimizer.

    components: ``int`` = ``None``
        Number of objective components for least-squares problems (pounders, dfols,
        ibcdfo). When set, an ``fvec`` field is added to the internal history.
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
        sample_points: Optional[npt.NDArray] = None,
        localopt_method: str = "scipy_Nelder-Mead",
        rk_const: Optional[float] = None,
        xtol_abs: Optional[float] = None,
        ftol_abs: Optional[float] = None,
        xtol_rel: Optional[float] = None,
        ftol_rel: Optional[float] = None,
        grtol: Optional[float] = None,
        gatol: Optional[float] = None,
        opt_return_codes: list[int] = [0],
        mu: float = 1e-8,
        nu: float = 1e-8,
        dist_to_bound_multiple: float = 0.5,
        random_seed: int = 1,
        stop_after_k_minima: Optional[int] = None,
        stop_after_k_runs: Optional[int] = None,
        periodic: bool = False,
        run_max_eval: Optional[int] = None,
        lhs_divisions: int = 0,
        scipy_kwargs: Optional[dict] = None,
        dfols_kwargs: Optional[dict] = None,
        components: Optional[int] = None,
        **kwargs,
    ) -> None:

        self.vocs = vocs

        gen_specs: Dict[str, Any] = {}
        gen_specs["user"] = {}

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
            "xtol_rel",
            "ftol_rel",
            "grtol",
            "gatol",
            "mu",
            "nu",
            "opt_return_codes",
            "dist_to_bound_multiple",
            "max_active_runs",
            "random_seed",
            "stop_after_k_minima",
            "stop_after_k_runs",
            "run_max_eval",
            "lhs_divisions",
            "scipy_kwargs",
            "dfols_kwargs",
            "components",
        ]

        for k in FIELDS:
            val = locals().get(k)
            if val is not None:
                gen_specs["user"][k] = val

        if periodic:
            gen_specs["user"]["periodic"] = True

        super().__init__(vocs, History, {}, gen_specs, {}, **kwargs)

        # APOSMM manages sim_id internally -- don't remap to _id
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
        if components is not None or "components" in gen_specs.get("user", {}):
            gen_specs["persis_in"].append("fvec")
        if localopt_method in ["LD_MMA", "blmvm", "scipy_BFGS"]:
            gen_specs["persis_in"].append("grad")

        # ---- Initialize APOSMM state using the canonical functions ----
        user_specs = gen_specs["user"]
        self._user_specs = user_specs
        self._max_active_runs = max_active_runs

        libE_info: Dict[str, Any] = {"comm": []}  # no comm needed in direct mode
        self._n, self._n_s, self._rk_const, self._ld, self._mu, self._nu, _, self._local_H = initialize_APOSMM(
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

        # Build reverse mapping: VOCS field name -> (internal_name, index, total_size)
        self._reverse_mapping: dict = {}
        for internal_name, vocs_names in self.variables_mapping.items():
            for i, vocs_name in enumerate(vocs_names):
                self._reverse_mapping[vocs_name] = (internal_name, i, len(vocs_names))

        # RNG
        self._rng = self.persis_info.get("rand_stream", np.random.default_rng(random_seed))
        self.persis_info["nworkers"] = max_active_runs

        # State tracking
        self.all_local_minima: List[npt.NDArray] = []
        self._told_initial_sample = False
        self._first_called_method: Optional[str] = None
        self._initial_sample_generated = False
        self._initial_suggest_idx = 0
        self._pending_results: Optional[npt.NDArray] = None
        self._first_pass = True
        self._n_r = 0  # number of results received in last ingest
        self._stopped = False

    def _map_to_internal(self, results: npt.NDArray) -> npt.NDArray:
        """Map VOCS-named structured array to internal APOSMM field names (x, x_on_cube, f, sim_id, grad, fvec)."""
        if results is None or len(results) == 0:
            return results
        # If already has internal names, return as-is
        if "x" in results.dtype.names and "f" in results.dtype.names:
            return results

        n_rows = len(results)

        # Build dtype for internal array
        internal_fields: list = []
        added: set = set()
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

    def _slot_in_data(self, results: npt.NDArray):
        """Slot ingested results into local_H during initial sample phase. Direct write, no buffer."""
        n_new = len(results)
        old_len = len(self._local_H)
        needed = self._n_s + n_new

        if needed > old_len:
            self._local_H.resize(needed, refcheck=False)
            initialize_dists_and_inds(self._local_H, needed - old_len)

        for i, row in enumerate(results):
            idx = self._n_s + i
            self._local_H["sim_id"][idx] = idx
            for name in results.dtype.names:
                if name == "sim_id":
                    continue
                if name in self._local_H.dtype.names:
                    self._local_H[name][idx] = row[name]
            self._local_H["sim_ended"][idx] = True

        self._n_s += n_new
        update_history_dist(self._local_H, self._n)

    def _build_output(self, indices):
        """Build the output numpy array from local_H indices."""
        if not indices:
            return np.zeros(0, dtype=self.gen_specs["out"])
        out_fields = [i[0] for i in self.gen_specs["out"]]
        return self._local_H[indices][out_fields].copy()

    def _check_stop_criteria(self) -> bool:
        """Check if APOSMM stopping criteria have been met."""
        user_specs = self._user_specs
        if np.sum(self._local_H["local_min"]) >= user_specs.get("stop_after_k_minima", np.inf):
            return True
        if len(self._ended_runs) >= user_specs.get("stop_after_k_runs", np.inf):
            return True
        return False

    # ---- Public API ----

    def suggest_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        out_fields = [i[0] for i in self.gen_specs["out"]]

        if self._stopped:
            return np.zeros(0, dtype=self.gen_specs["out"])

        if self._first_called_method is None:
            self._first_called_method = "suggest"

        # Initial sample phase: generate random points once, return in batches
        if not self._told_initial_sample:
            if not self._initial_sample_generated:
                total = self._user_specs["initial_sample_size"]
                add_k_sample_points_to_local_H(
                    total,
                    self._user_specs,
                    self.persis_info,
                    self._n,
                    [],
                    self._local_H,
                    self._sim_id_to_child_inds,
                    self._rng,
                )
                self._initial_sample_generated = True
                self._initial_suggest_idx = 0

            k = num_points if num_points > 0 else (self._user_specs["initial_sample_size"] - self._initial_suggest_idx)
            start = self._initial_suggest_idx
            end = min(start + k, self._user_specs["initial_sample_size"])
            result = self._local_H[start:end][out_fields].copy()
            self._initial_suggest_idx = end
            return unmap_numpy_array(result, self.variables_mapping)

        # ---- Main optimization phase ----
        new_opt_inds = []
        new_inds = []

        # Process any pending ingested results through local optimizers
        if self._pending_results is not None:
            calc_in = self._pending_results
            self._pending_results = None

            # Update local_H with received results
            for row in calc_in:
                sim_id = int(row["sim_id"])
                self._local_H[sim_id]["sim_ended"] = True
                for name in calc_in.dtype.names:
                    if name in self._local_H.dtype.names:
                        self._local_H[name][sim_id] = row[name]
            self._n_s = int(np.sum(~self._local_H["local_pt"][: len(self._local_H)]))
            update_history_dist(self._local_H, self._n)

            # Feed results to active local optimizers
            for row in calc_in:
                sim_id = int(row["sim_id"])
                if self._sim_id_to_child_inds.get(sim_id):
                    for child_idx in list(self._sim_id_to_child_inds[sim_id]):
                        if child_idx not in self._local_opters:
                            continue
                        x_new = self._local_opters[child_idx].iterate(row[self._fields_to_pass])
                        if isinstance(x_new, ConvergedMsg):
                            x_opt = x_new.x
                            opt_flag = x_new.opt_flag
                            opt_ind = update_history_optimal(x_opt, opt_flag, self._local_H, self._run_order[child_idx])
                            new_opt_inds.append(opt_ind)
                            self._local_opters.pop(child_idx)
                            self._ended_runs.append(child_idx)
                        else:
                            add_to_local_H(self._local_H, x_new, self._user_specs, local_flag=1, on_cube=True)
                            new_inds.append(len(self._local_H) - 1)
                            self._run_order[child_idx].append(self._local_H[-1]["sim_id"])
                            self._run_pts[child_idx].append(x_new)
                            sid = self._local_H[-1]["sim_id"]
                            if sid in self._sim_id_to_child_inds:
                                self._sim_id_to_child_inds[sid] += (child_idx,)
                            else:
                                self._sim_id_to_child_inds[sid] = (child_idx,)

        # Decide where to start new local optimization runs
        starting_inds = decide_where_to_start_localopt(
            self._local_H, self._n, self._n_s, self._rk_const, self._ld, self._mu, self._nu
        )

        for ind in starting_inds:
            if len([p for p in self._local_opters.values() if p.is_running]) >= self._max_active_runs:
                break

            self._local_H["started_run"][ind] = 1

            local_opter = LocalOptInterfacer(
                self._user_specs,
                self._local_H[ind]["x_on_cube"],
                self._local_H[ind]["f"] if "f" in self._fields_to_pass else self._local_H[ind]["fvec"],
                self._local_H[ind]["grad"] if "grad" in self._fields_to_pass else None,
            )

            self._local_opters[self._total_runs] = local_opter

            x_new = local_opter.iterate(self._local_H[ind][self._fields_to_pass])

            add_to_local_H(self._local_H, x_new, self._user_specs, local_flag=1, on_cube=True)
            new_inds.append(len(self._local_H) - 1)

            self._run_order[self._total_runs] = [ind, self._local_H[-1]["sim_id"]]
            self._run_pts[self._total_runs] = [self._local_H["x_on_cube"], x_new]

            sid = self._local_H[-1]["sim_id"]
            if sid in self._sim_id_to_child_inds:
                self._sim_id_to_child_inds[sid] += (self._total_runs,)
            else:
                self._sim_id_to_child_inds[sid] = (self._total_runs,)

            self._total_runs += 1

        # Fill remaining slots with sample points (mirrors original aposmm logic)
        if self._first_pass:
            num_samples = self._max_active_runs - 1 - len(new_inds)
            self._first_pass = False
        else:
            num_samples = self._n_r - len(new_inds)

        if num_samples > 0:
            prev_len = len(self._local_H)
            add_k_sample_points_to_local_H(
                num_samples,
                self._user_specs,
                self.persis_info,
                self._n,
                [],
                self._local_H,
                self._sim_id_to_child_inds,
                self._rng,
            )
            new_inds = new_inds + list(range(prev_len, len(self._local_H)))

        all_inds = new_inds + new_opt_inds

        if len(all_inds) == 0:
            return np.zeros(0, dtype=[(name, self._local_H.dtype[name]) for name in out_fields])

        result = self._local_H[all_inds][out_fields].copy()

        # Track local minima for suggest_updates()
        if result["local_min"].any():
            min_idxs = result["local_min"]
            self.all_local_minima.append(result[min_idxs].copy())

        return unmap_numpy_array(result, self.variables_mapping)

    def ingest_numpy(self, results: npt.NDArray, tag: int = 0) -> None:
        """Send the results of evaluations to the generator, as a NumPy array."""

        if results is None:
            return

        if self._stopped:
            return

        if self._first_called_method is None:
            self._first_called_method = "ingest"

        # Map from VOCS field names to internal APOSMM names
        results = self._map_to_internal(results)

        if not self._told_initial_sample:
            # Initial sample phase: write directly into local_H
            self._slot_in_data(results)
            if self._n_s >= self._user_specs["initial_sample_size"]:
                self._told_initial_sample = True
            return

        # Main phase: buffer results for processing in next suggest call
        self._n_r = len(results)
        self._pending_results = results.copy()

        if self._check_stop_criteria():
            self._stopped = True
            self._clean_up()

    def _clean_up(self):
        """Destroy all running local optimizers."""
        for _i, p in list(self._local_opters.items()):
            p.destroy()
        self._local_opters.clear()

    def finalize(self) -> None:
        """Stop the generator and clean up local optimizer processes."""
        self._clean_up()
        self._stopped = True

    def export(self, vocs_field_names: bool = False, as_dicts: bool = False) -> tuple:
        """Return the generator's results.

        Parameters
        ----------
        vocs_field_names : bool, optional
            If True, return local_H with variables unmapped from arrays back to individual fields.
        as_dicts : bool, optional
            If True, return local_H as list of dictionaries instead of numpy array.

        Returns
        -------
        local_H : npt.NDArray | list | None
            Generator history array.
        persis_info : dict | None
            Persistent information including run_order.
        tag : int | None
            Status flag.
        """
        from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
        from libensemble.utils.misc import np_to_list_dicts

        if len(self._local_H) == 0:
            return (None, None, None)

        local_H = self._local_H.copy()
        persis_info = dict(self.persis_info)
        persis_info["run_order"] = self._run_order

        if vocs_field_names and self.variables_mapping:
            local_H = unmap_numpy_array(local_H, self.variables_mapping)
        if as_dicts:
            if vocs_field_names and self.variables_mapping:
                local_H = np_to_list_dicts(local_H, self.variables_mapping)
            else:
                local_H = np_to_list_dicts(local_H)
        return (local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG)

    def suggest_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima


# Backward-compatible alias
APOSMMDirect = APOSMM
