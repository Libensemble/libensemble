"""Run a small, reproducible Latin-hypercube sampling workflow locally.

Run with ``python local_sampling.py`` from a fresh working directory. The run
writes ``local_sampling.npy`` and ``local_sampling.pickle`` in that directory.
"""

import numpy as np
from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes.sampling import LatinHypercubeSample
from libensemble.specs import GenSpecs, LibeSpecs, SimSpecs


def simulate(inputs: dict, **kwargs) -> dict:
    """Evaluate a simple objective using the two sampled variables."""
    x0 = inputs["x0"]
    x1 = inputs["x1"]
    return {"f": x0**2 + x1**2}


if __name__ == "__main__":
    vocs = VOCS(
        variables={"x0": [-3.0, 3.0], "x1": [-2.0, 2.0]},
        objectives={"f": "MINIMIZE"},
    )

    # Keep the complete LHS in one initial batch. Splitting it across requests
    # would create multiple smaller designs instead of one 12-point design.
    generator = LatinHypercubeSample(vocs, random_seed=1)
    ensemble = Ensemble(
        sim_specs=SimSpecs(simulator=simulate, vocs=vocs),
        gen_specs=GenSpecs(
            generator=generator,
            vocs=vocs,
            initial_batch_size=12,
            batch_size=12,
        ),
        # Using local workers lets this example run with plain Python. The default
        # ensemble directory is created in the fresh working directory.
        libE_specs=LibeSpecs(comms="local", nworkers=4, safe_mode=True),
    )

    history, _, exit_flag = ensemble.run(sim_max=12)

    if ensemble.is_manager:
        completed = history[history["sim_ended"]]
        finite = completed[np.isfinite(completed["f"])]
        history_path = ensemble.save_output("local_sampling", append_attrs=False)
        print(f"Saved History to {history_path}")
        print(f"Completed {len(completed)} of 12 evaluations; exit flag: {exit_flag}")
        if len(finite):
            best = finite[np.argmin(finite["f"])]
            print(f"Best point: x0={best['x0']:.4f}, x1={best['x1']:.4f}, f={best['f']:.6f}")
        if exit_flag != 0:
            raise RuntimeError(f"libEnsemble exited with flag {exit_flag}")
