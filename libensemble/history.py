import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from libensemble.tools.fields_keys import libE_fields, protected_libE_fields

if TYPE_CHECKING:
    from libensemble.logger import LibensembleLogger

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    assert isinstance(logger, LibensembleLogger)


# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


class History:
    """The History class provides methods for managing the history array.

    **Object Attributes:**

    These are set on initialization.

    :ivar numpy.ndarray H:
        History array storing rows for each point. Field names are in
        libensemble/tools/fields_keys.py. Numpy structured array.

    :ivar int offset:
        Starting index for this ensemble (after H0 read in)

    :ivar int index:
        Index where libEnsemble should start filling in H

    :ivar int sim_started_count:
        Number of points given to sim functions (according to H)

    :ivar int sim_ended_count:
        Number of points evaluated  (according to H)

    Note that index, sim_started_count and sim_ended_count reflect the total number of points
    in H and therefore include those prepended to H in addition to the current run.

    """

    def __init__(
        self, alloc_specs: dict, sim_specs: dict, gen_specs: dict, exit_criteria: dict, H0: npt.NDArray
    ) -> None:
        """
        Forms the numpy structured array that records everything from the
        libEnsemble run

        """
        L = exit_criteria.get("sim_max", 100)

        # Combine all 'out' fields (if they exist) in sim_specs, gen_specs, alloc_specs
        specs = [sim_specs, gen_specs, alloc_specs]
        specs_dtype_list = list(set(libE_fields + sum([k.get("out", []) for k in specs if k], [])))

        if len(H0):
            # remove duplicate fields from specs dtype list if those already in H0 (H0 takes precedence)
            pruned_specs_dtype_list = [i for i in specs_dtype_list if i[0] not in H0.dtype.names]
            H_fields = list(set(pruned_specs_dtype_list + H0.dtype.descr))

            H = np.zeros(L + len(H0), dtype=H_fields)

            # Prepend H with H0
            fields = H0.dtype.names

            for field in fields:
                H[field][: len(H0)] = H0[field]

            if "sim_started" not in fields:
                logger.manager_warning(  # type: ignore[attr-defined]
                    "Marking entries in H0 as having been 'sim_started' and 'sim_ended'"
                )

                H["sim_started"][: len(H0)] = 1
                H["sim_ended"][: len(H0)] = 1
            elif "sim_ended" not in fields:
                logger.manager_warning(  # type: ignore[attr-defined]
                    "Marking entries in H0 as having been 'sim_ended' if 'sim_started'"
                )

                H["sim_ended"][: len(H0)] = H0["sim_started"]

            if "sim_id" not in fields:
                logger.manager_warning("Assigning sim_ids to entries in H0")  # type: ignore[attr-defined]

                H["sim_id"][: len(H0)] = np.arange(0, len(H0))
        else:
            H = np.zeros(L + len(H0), dtype=specs_dtype_list)

        H["sim_id"][-L:] = -1
        if "_id" in H.dtype.names:
            H["_id"][-L:] = -1
        H["sim_started_time"][-L:] = np.inf
        H["gen_informed_time"][-L:] = np.inf

        if "resource_sets" in H.dtype.names:
            H["resource_sets"][-L:] = 1

        self.H = H
        self.using_H0 = len(H0) > 0
        self.index = len(H0)
        self.grow_count = 0
        self.safe_mode = False
        self.use_cache = False

        self.sim_started_count: int = np.sum(H["sim_started"])
        self.sim_ended_count: int = np.sum(H["sim_ended"])
        self.gen_informed_count: int = np.sum(H["gen_informed"])
        self.given_back_warned = False

        self.sim_started_offset: int = self.sim_started_count
        self.sim_ended_offset: int = self.sim_ended_count
        self.gen_informed_offset: int = self.gen_informed_count

        self.last_started = -1
        self.last_ended = -1

    def init_cache(
        self,
        cache_name: str,
        cache_dir: str | Path,
        spec_hash: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self.cache_dir / Path(cache_name + ".npy")
        self.cache_meta = self.cache_dir / Path(cache_name + ".meta.json")
        self.spec_hash = spec_hash
        self.use_cache = True
        self.cache_set = False

        # Precompute the sorted user-field names and their dtypes once, so
        # _shelf_longrunning_sims doesn't recompute them on every sim return.
        libE_field_names = {k[0] for k in libE_fields}
        self.cache_keys = sorted([n for n in self.H.dtype.names if n not in libE_field_names])
        self.cache_dtype = np.dtype(sorted([(n, self.H.dtype.fields[n][0]) for n in self.cache_keys]))

        # Buffer for new entries collected during this run; deduplicated via bytes key.
        self._cache_buffer: list = []
        self._cache_seen: set = set()

        # Validate any existing cache against the configuration hash.
        cache_valid = False
        if self.cache.exists():
            if self.cache_meta.exists():
                try:
                    with open(self.cache_meta) as f:
                        meta = json.load(f)
                    if meta.get("spec_hash") == spec_hash:
                        cache_valid = True
                except (json.JSONDecodeError, KeyError):
                    pass
            if not cache_valid:
                logger.debug(
                    "Cache hash mismatch or missing metadata — starting fresh: %s",
                    self.cache.name,
                )
                self.cache.unlink(missing_ok=True)

        if not self.cache.exists():
            self.cache.touch()

        try:
            self.in_cache = np.load(self.cache, allow_pickle=True)
        except EOFError:
            self.in_cache = None

        # Pre-populate the seen-set from any on-disk entries so we don't re-add them.
        # Also mark cache_set=True immediately when there is existing data — the manager
        # uses this flag to decide whether to scan the cache when dispatching sim work.
        if self.in_cache is not None and len(self.in_cache) > 0:
            for row in self.in_cache:
                self._cache_seen.add(row.tobytes())
            self.cache_set = True

    def _append_new_fields(self, H_f: npt.NDArray) -> None:
        import numpy.lib.recfunctions as rfn

        dtype_new: np.dtype = np.dtype(list(set(self.H.dtype.descr + rfn.repack_fields(H_f).dtype.descr)))

        H_new = np.zeros(len(self.H), dtype=dtype_new)
        old_fields = self.H.dtype.names
        for field in old_fields:
            H_new[field][: len(self.H)] = self.H[field]
        self.H = H_new

    def _shelf_longrunning_sims(self, index):
        """Cache any f values that ran for more than a second.

        Uses a bytes-keyed set for O(1) deduplication instead of np.unique on
        every insertion, and accumulates new entries in a plain Python list that
        is only materialised into a structured array at save_cache() time.
        """
        if self.H[index]["sim_ended_time"] - self.H[index]["sim_started_time"] <= 1:
            return
        entry = np.array([self.H[index][self.cache_keys]], dtype=self.cache_dtype)
        key = entry[0].tobytes()
        if key in self._cache_seen:
            return
        self._cache_seen.add(key)
        self._cache_buffer.append(entry)
        self.cache_set = True

    def _materialize_cache(self) -> npt.NDArray | None:
        """Combine the on-disk cache with any buffered new entries into one array."""
        parts = []
        if self.in_cache is not None:
            parts.append(self.in_cache)
        if self._cache_buffer:
            parts.append(np.concatenate(self._cache_buffer))
        if not parts:
            return None
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def save_cache(self) -> None:
        if self.use_cache and self.cache_set:
            combined = self._materialize_cache()
            if combined is not None:
                np.save(self.cache, combined, allow_pickle=True)
                if self.spec_hash:
                    with open(self.cache_meta, "w") as f:
                        json.dump({"spec_hash": self.spec_hash}, f)

    def get_shelved_sims(self) -> npt.NDArray:
        combined = self._materialize_cache()
        return combined if combined is not None else np.load(self.cache, allow_pickle=True)

    @staticmethod
    def _classify_fields(fields, returned_H, H):
        """Partition returned fields into three buckets for update_history_f.

        Returns
        -------
        scalar_fields : list[str]
            Fields whose per-row value is a scalar or object (can be assigned
            with a single fancy-indexed write across all rows).
        uniform_fields : list[str]
            Fixed-shape array fields whose shape exactly matches H's storage
            shape (can also be assigned in one fancy-indexed write).
        ragged_fields : list[str]
            Fixed-shape array fields that are *smaller* than H's storage shape
            (need per-row slice assignment).
        """
        scalar_fields = []
        uniform_fields = []
        ragged_fields = []
        for field in fields:
            if field in protected_libE_fields:
                continue
            dt = returned_H.dtype[field]
            if dt.shape == () or dt.hasobject:
                scalar_fields.append(field)
            else:
                # Compare element shape: returned vs H's allocated shape
                h_shape = H.dtype[field].shape
                r_shape = dt.shape
                if r_shape == h_shape:
                    uniform_fields.append(field)
                else:
                    ragged_fields.append(field)
        return scalar_fields, uniform_fields, ragged_fields

    def update_history_f(self, D: dict, kill_canceled_sims: bool = False) -> None:
        """
        Updates the history after points have been evaluated
        """

        new_inds = D["libE_info"]["H_rows"]
        returned_H = D["calc_out"]

        fields = returned_H.dtype.names if returned_H is not None else []
        if returned_H is not None and any([field not in self.H.dtype.names for field in returned_H.dtype.names]):
            self._append_new_fields(returned_H)

        if self.safe_mode:
            for field in fields:
                assert field not in protected_libE_fields, "The field '" + field + "' is protected"

        new_inds = np.asarray(new_inds)

        if fields and returned_H is not None:
            scalar_fields, uniform_fields, ragged_fields = self._classify_fields(fields, returned_H, self.H)

            # Vectorized assignment for scalar and object fields (one op per field)
            for field in scalar_fields:
                self.H[field][new_inds] = returned_H[field]

            # Vectorized assignment for fixed-shape array fields that exactly match H's shape
            for field in uniform_fields:
                self.H[field][new_inds] = returned_H[field]

            # Per-row loop only for ragged (partial-fill) array fields
            for j, ind in enumerate(new_inds):
                for field in ragged_fields:
                    H0_size = len(returned_H[field][j])
                    assert H0_size <= len(self.H[field][ind]), (
                        "History update Error: Too many values received for " + field
                    )
                    assert H0_size, "History update Error: No values in this field " + field
                    self.H[field][ind][:H0_size] = returned_H[field][j]

        # Batch-update bookkeeping fields for all returned rows at once
        t = time.time()
        self.H["sim_ended"][new_inds] = True
        self.H["sim_ended_time"][new_inds] = t
        self.sim_ended_count += len(new_inds)

        if self.use_cache:
            for ind in new_inds:
                self._shelf_longrunning_sims(ind)

        if kill_canceled_sims:
            for j in range(self.last_ended + 1, np.max(new_inds) + 1):
                if self.H["sim_ended"][j]:
                    self.last_ended += 1
                else:
                    break

    def update_history_x_out(self, q_inds: npt.NDArray, sim_worker: int, kill_canceled_sims: bool = False) -> None:
        """
        Updates the history (in place) when new points have been given out to be evaluated

        Parameters
        ----------
        q_inds: numpy.typing.NDArray
            Row IDs for history array H

        sim_worker: int
            Worker ID
        """
        q_inds = np.atleast_1d(q_inds)
        t = time.time()

        self.H["sim_started"][q_inds] = True
        self.H["sim_started_time"][q_inds] = t
        self.H["sim_worker"][q_inds] = sim_worker

        self.sim_started_count += len(q_inds)
        if kill_canceled_sims:
            self.last_started = np.max(q_inds)

    def update_history_to_gen(self, q_inds: npt.NDArray):
        """Updates the history (in place) when points are given back to the gen"""
        q_inds = np.atleast_1d(q_inds)
        t = time.time()

        if q_inds.size > 0:
            if np.all(self.H["sim_ended"][q_inds]):
                self.H["gen_informed"][q_inds] = True

            elif np.any(self.H["sim_ended"][q_inds]):  # sporadic returned points need updating
                for ind in q_inds[self.H["sim_ended"][q_inds]]:
                    self.H["gen_informed"][ind] = True

            if self.using_H0 and not self.given_back_warned:
                logger.manager_warning(  # type: ignore[attr-defined]
                    "Giving entries in H0 back to gen. Marking entries in H0 as 'gen_informed' if 'sim_ended'."
                )

                self.given_back_warned = True

            self.H["gen_informed_time"][q_inds] = t
            self.gen_informed_count += len(q_inds)

    def update_history_x_in(self, gen_worker: int, D: npt.NDArray, gen_started_time: int) -> None:
        """
        Updates the history (in place) when new points have been returned from a gen

        Parameters
        ----------
        gen_worker: int
            The worker who generated these points
        D: numpy.typing.NDArray
            Output from gen_func
        """

        if len(D) == 0:
            return

        if any([field not in self.H.dtype.names for field in D.dtype.names]):
            self._append_new_fields(D)

        t = time.time()
        rows_remaining = len(self.H) - self.index

        if "sim_id" not in D.dtype.names:
            # gen method must not be adjusting sim_id, just append to self.H
            num_new = len(D)

            if num_new > rows_remaining:
                self.grow_count = max(num_new - rows_remaining, 2 * self.grow_count)
                self.grow_H(self.grow_count)

            update_inds = np.arange(self.index, self.index + num_new)
            self.H["sim_id"][self.index : self.index + num_new] = range(self.index, self.index + num_new)
        else:
            # gen method is building sim_id or adjusting values in existing sim_id rows.

            # Ensure there aren't any gaps in the generated sim_id values:
            assert np.all(
                np.isin(np.arange(self.index, np.max(D["sim_id"]) + 1), D["sim_id"])
            ), "The generator function has produced sim_ids that are not in order."

            num_new = len(np.setdiff1d(D["sim_id"], self.H["sim_id"]))

            if num_new > rows_remaining:
                self.grow_count = max(num_new - rows_remaining, 2 * self.grow_count)
                self.grow_H(self.grow_count)

            update_inds = D["sim_id"]

        for field in D.dtype.names:
            if field in protected_libE_fields:
                if self.safe_mode:
                    assert False, "The field '" + field + "' is protected"
                continue

            self.H[field][update_inds] = D[field]

        first_gen_inds = update_inds[self.H["gen_ended_time"][update_inds] == 0]
        self.H["gen_started_time"][first_gen_inds] = gen_started_time
        self.H["gen_ended_time"][first_gen_inds] = t
        self.H["gen_worker"][first_gen_inds] = gen_worker
        self.index += num_new

    def grow_H(self, k: int) -> None:
        """
        Adds k rows to H in response to gen_f producing more points than
        available rows in H.

        Parameters
        ----------
        k: int
            Number of rows to add to H
        """
        H_1 = np.zeros(k, dtype=self.H.dtype)
        H_1["sim_id"] = -1
        if "_id" in H_1.dtype.names:
            H_1["_id"] = -1
        H_1["sim_started_time"] = np.inf
        H_1["gen_informed_time"] = np.inf
        if "resource_sets" in H_1.dtype.names:
            H_1["resource_sets"] = 1
        self.H = np.append(self.H, H_1)

    # Could be arguments here to return different truncations eg. all done, given etc...
    def trim_H(self) -> npt.NDArray:
        """Returns truncated array"""
        return self.H[: self.index]
