"""Evaluate a fixed list of points with the standardized preloaded generator.

Run with ``python preloaded_points.py`` from a fresh working directory. The
run writes ``preloaded_points.npy`` and ``preloaded_points.pickle`` there.
"""

import numpy as np
from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes.preloaded import PreloadedSampleGenerator
from libensemble.specs import GenSpecs, LibeSpecs, SimSpecs


def simulate(inputs: dict, **kwargs) -> dict:
    """Return the objective for one point supplied by the preloaded generator."""
    x0 = inputs["x0"]
    x1 = inputs["x1"]
    return {"f": x0**2 + x1**2}


if __name__ == "__main__":
    # Every point must provide the variable fields the simulator reads. Constants,
    # if present in VOCS, should also be included in each point or added explicitly.
    points = [
        {"x0": -1.0, "x1": 0.5},
        {"x0": 0.0, "x1": 0.0},
        {"x0": 1.0, "x1": 0.5},
    ]
    vocs = VOCS(
        variables={"x0": [-3.0, 3.0], "x1": [-2.0, 2.0]},
        objectives={"f": "MINIMIZE"},
    )

    generator = PreloadedSampleGenerator(points, vocs=vocs, batch_size=2)
    ensemble = Ensemble(
        sim_specs=SimSpecs(simulator=simulate, vocs=vocs),
        gen_specs=GenSpecs(generator=generator, vocs=vocs),
        libE_specs=LibeSpecs(comms="local", nworkers=2, safe_mode=True),
    )

    history, _, exit_flag = ensemble.run(sim_max=len(points))

    if ensemble.is_manager:
        completed = history[history["sim_ended"]]
        finite = completed[np.isfinite(completed["f"])]
        history_path = ensemble.save_output("preloaded_points", append_attrs=False)
        print(f"Saved History to {history_path}")
        print(f"Evaluated {len(completed)} of {len(points)} supplied points")
        if len(finite):
            best = finite[np.argmin(finite["f"])]
            print(f"Lowest objective: x0={best['x0']:.4f}, x1={best['x1']:.4f}, f={best['f']:.6f}")
        if exit_flag != 0:
            raise RuntimeError(f"libEnsemble exited with flag {exit_flag}")
