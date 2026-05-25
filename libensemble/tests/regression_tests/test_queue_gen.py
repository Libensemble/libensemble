"""Test for QueueGenerator / QueueService

Uses QueueService to spin up libE in a thread with a trivial doubler sim,
submits N work items, drains results, shuts down, joins.


Run with:
    python test_queue_gen.py
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4

import math
import time

import numpy as np
from gest_api.vocs import VOCS

from libensemble.gen_classes.queue_gen import QueueService
from libensemble.specs import ExitCriteria, LibeSpecs, SimSpecs

NWORKERS = 4
N_SUBMITS = 10


def doubler_sim(InputArray, _, sim_specs):
    """Trivial sim: returns 2*x. Sleeps a bit to mimic real work."""
    time.sleep(0.5)
    out = np.zeros(1, dtype=sim_specs["out"])
    out["y"] = 2.0 * InputArray["x"][0]
    return out


def main():
    vocs = VOCS(variables={"x": [-100.0, 100.0]}, objectives={"y": "MINIMIZE"})
    sim_specs = SimSpecs(sim_f=doubler_sim, inputs=["x"], outputs=[("y", float)])
    libE_specs = LibeSpecs(nworkers=NWORKERS, service_mode=True, final_gen_send=True)
    exit_criteria = ExitCriteria(sim_max=N_SUBMITS)

    service = QueueService(vocs, sim_specs, libE_specs, exit_criteria)
    service.start()
    print(f"libE thread started ({NWORKERS} workers)")

    # Submit work (1, 2, 3....), then  signal shutdown.
    for i in range(N_SUBMITS):
        service.submit({"x": float(i)})
    service.shutdown()
    print(f"submitted {N_SUBMITS} items + shutdown sentinel")

    # Block-drain until all results collected (or timeout)
    results = service.collect_results(N_SUBMITS, timeout=60)
    print(f"\ncollected {len(results)}/{N_SUBMITS}")

    # Verify y == 2*x
    ok = all(math.isclose(r["y"], 2 * r["x"]) for r in results)
    print("PASS" if ok else "FAIL")

    # Wait for libE to wind down
    service.join(timeout=10)
    print("done")


if __name__ == "__main__":
    main()
