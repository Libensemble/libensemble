"""Test for QueueGenerator / QueueService — streaming with replacement.

Submit NWORKERS items so each worker has work, then as each result comes
back submit a replacement until TOTAL items have been submitted. Drain
the remaining results and shutdown.

Run with:
    python test_queue_gen_streaming.py
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
TOTAL = 16  # total items to submit (>> NWORKERS so the rolling window matters)


def doubler_sim(InputArray, _, sim_specs):
    """Trivial sim: returns 2*x. Sleeps a bit to mimic real work."""
    time.sleep(0.5)
    out = np.zeros(1, dtype=sim_specs["out"])
    out["y"] = 2.0 * InputArray["x"][0]
    return out


def main():
    vocs = VOCS(variables={"x": [-100.0, 100.0]}, objectives={"y": "MINIMIZE"})
    sim_specs = SimSpecs(sim_f=doubler_sim, inputs=["x"], outputs=[("y", float)])
    libE_specs = LibeSpecs(nworkers=NWORKERS, service_mode=True,
                           service_mode_idle_timeout=30, final_gen_send=True)
    exit_criteria = ExitCriteria(sim_max=TOTAL)

    service = QueueService(vocs, sim_specs, libE_specs, exit_criteria)
    service.start()
    print(f"libE thread started ({NWORKERS} workers)")

    # Submit one item per worker
    for i in range(NWORKERS):
        service.submit({"x": float(i)})
    submitted = NWORKERS
    print(f"submitted initial {NWORKERS} items")

    # As each result arrives, submit a replacement until TOTAL submitted
    results = []
    for r in service.stream_results(TOTAL, timeout=60):
        results.append(r)
        print(f"  got ({len(results)}/{TOTAL}): {r}")
        if submitted < TOTAL:
            service.submit({"x": float(submitted)})
            submitted += 1

    service.shutdown()
    print(f"\nsubmitted {submitted}, collected {len(results)}/{TOTAL}")

    ok = all(math.isclose(r["y"], 2 * r["x"]) for r in results)
    print("PASS" if ok else "FAIL")

    service.join(timeout=10)
    print("done")


if __name__ == "__main__":
    main()
