"""Generator class for evaluating a pre-existing sample of points.

This module provides :class:`PreloadedSampleGenerator`, a gest-api compatible
generator that wraps a user-supplied set of points and serves them to libEnsemble
workers for simulation evaluation.  No VOCS or online generation is required.

Typical usage::

    import numpy as np
    from libensemble import Ensemble
    from libensemble.gen_classes.preloaded import PreloadedSampleGenerator
    from libensemble.specs import GenSpecs, SimSpecs

    H0 = np.zeros(500, dtype=[("x", float, 8), ("sim_id", int)])
    H0["x"] = my_existing_points
    H0["sim_id"] = range(500)

    sampling = Ensemble(parse_args=True)
    sampling.gen_specs = GenSpecs(generator=PreloadedSampleGenerator(H0))
    sampling.sim_specs = SimSpecs(sim_f=my_sim, inputs=["x"], out=[("f", float)])
    sampling.run(sim_max=len(H0))

This replaces the legacy ``give_pregenerated_work`` allocator pattern, which
required a custom ``AllocSpecs`` and bypassed the generator entirely.  With
:class:`PreloadedSampleGenerator` the default ``only_persistent_gens`` allocator
is used transparently.
"""

from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from gest_api import Generator
from gest_api.vocs import VOCS

from libensemble.utils.misc import np_to_list_dicts

__all__ = ["PreloadedSampleGenerator"]

# Sentinel VOCS used when the caller does not supply one.  The generator never
# actually samples from this; it is required only to satisfy the gest-api
# Generator base class constructor.  The list form ``[low, high]`` is the
# canonical gest-api shorthand for a continuous variable.
_SENTINEL_VOCS = VOCS(variables={"_preloaded": [0.0, 1.0]})


class PreloadedSampleGenerator(Generator):
    """A gest-api generator that serves a fixed, pre-existing set of points.

    Points are taken from a numpy structured array (or a list of dicts) supplied
    at construction time and returned to libEnsemble in chunks via :meth:`suggest`.
    Once all points are exhausted :meth:`suggest` returns an empty list, which
    causes the default ``only_persistent_gens`` allocator to shut down the
    generator and end the ensemble.

    No online learning or VOCS sampling is performed; :meth:`ingest` is a no-op.

    Parameters
    ----------
    points:
        Pre-generated points to evaluate.  May be either:

        * A numpy structured array whose field names match ``sim_specs["in"]``.
          Fields ``sim_id`` and ``sim_started`` are ignored (libEnsemble manages
          them internally).
        * A list of dicts with consistent keys.
    vocs:
        Optional VOCS object.  If omitted a sentinel placeholder is used — the
        generator does not sample from it.
    batch_size:
        Number of points to return per :meth:`suggest` call.  Defaults to
        returning all remaining points at once (i.e. one large batch).  Setting
        this to a positive integer enables streaming delivery, which can reduce
        peak memory pressure for very large pre-generated samples.

    Examples
    --------
    Evaluate 1000 pre-generated borehole inputs:

    .. code-block:: python

        import numpy as np
        from libensemble import Ensemble
        from libensemble.gen_classes.preloaded import PreloadedSampleGenerator
        from libensemble.sim_funcs.borehole import borehole as sim_f, gen_borehole_input
        from libensemble.specs import GenSpecs, SimSpecs

        n_samp = 1000
        pts = np.zeros(n_samp, dtype=[("x", float, 8)])
        pts["x"] = gen_borehole_input(n_samp)

        sampling = Ensemble(parse_args=True)
        sampling.gen_specs = GenSpecs(generator=PreloadedSampleGenerator(pts))
        sampling.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], out=[("f", float)])
        sampling.run(sim_max=n_samp)
    """

    def __init__(
        self,
        points: Union[npt.NDArray, List[dict]],
        vocs: Optional[VOCS] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(vocs if vocs is not None else _SENTINEL_VOCS)

        # Normalise to a list-of-dicts for the gest-api suggest() return type.
        if isinstance(points, np.ndarray):
            # Strip fields that libEnsemble manages internally before converting.
            internal = {"sim_id", "sim_started", "sim_ended", "gen_worker"}
            user_fields = [n for n in points.dtype.names if n not in internal]
            self._points: List[dict] = np_to_list_dicts(points[user_fields])
        else:
            self._points = list(points)

        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self._batch_size = batch_size
        self._cursor: int = 0

    # ------------------------------------------------------------------
    # gest-api interface
    # ------------------------------------------------------------------

    def _validate_vocs(self, vocs: VOCS) -> None:  # noqa: D102
        pass  # No VOCS constraints — pre-loaded points already exist.

    def suggest(self, n_trials: int) -> List[dict]:
        """Return the next batch of pre-loaded points.

        Parameters
        ----------
        n_trials:
            Hint from the allocator for how many points are needed.  When
            ``batch_size`` was set at construction that value takes precedence;
            otherwise ``n_trials`` is honoured.

        Returns
        -------
        list[dict]
            Next chunk of points, or ``[]`` when all points have been served.
        """
        if self._cursor >= len(self._points):
            return []

        # Determine how many points to emit this call.
        chunk = self._batch_size if self._batch_size is not None else n_trials
        # Never return more than what is left.
        chunk = min(chunk, len(self._points) - self._cursor)

        batch = self._points[self._cursor : self._cursor + chunk]
        self._cursor += chunk
        return batch

    def ingest(self, calc_in: List[dict]) -> None:  # noqa: D102 — intentional no-op
        """Receive simulation results (no-op — pre-loaded sample needs no feedback)."""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_remaining(self) -> int:
        """Number of points not yet served by :meth:`suggest`."""
        return max(0, len(self._points) - self._cursor)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"total={len(self._points)}, "
            f"served={self._cursor}, "
            f"remaining={self.n_remaining}, "
            f"batch_size={self._batch_size!r})"
        )
