"""Test for service_mode_idle_timeout.

Producer stays silent after a single submit. With service_mode_idle_timeout
set, libE should exit by itself after the timeout instead of hanging.

Run with:
    python test_queue_gen_idle_timeout.py
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 2

import time

import numpy as np
from gest_api.vocs import VOCS

from libensemble.gen_classes.queue_gen import QueueService
from libensemble.specs import ExitCriteria, LibeSpecs, SimSpecs

NWORKERS = 2
IDLE_TIMEOUT = 3.0  # seconds


def doubler_sim(InputArray, _, sim_specs):
    out = np.zeros(1, dtype=sim_specs["out"])
    out["y"] = 2.0 * InputArray["x"][0]
    return out


def main():
    vocs = VOCS(variables={"x": [-100.0, 100.0]}, objectives={"y": "MINIMIZE"})
    sim_specs = SimSpecs(sim_f=doubler_sim, inputs=["x"], outputs=[("y", float)])
    libE_specs = LibeSpecs(
        nworkers=NWORKERS,
        service_mode=True,
        service_mode_idle_timeout=IDLE_TIMEOUT,
        final_gen_send=True,
    )
    exit_criteria = ExitCriteria(sim_max=1000)  # large so idle_timeout fires first

    service = QueueService(vocs, sim_specs, libE_specs, exit_criteria)
    service.start()
    print(f"libE thread started ({NWORKERS} workers), idle_timeout={IDLE_TIMEOUT}s")

    service.submit({"x": 1.0})
    print("submitted 1 item, now going silent")

    t0 = time.time()
    service.join(timeout=IDLE_TIMEOUT + 10)
    elapsed = time.time() - t0
    print(f"libE thread exited after {elapsed:.2f}s (expected ~{IDLE_TIMEOUT}s)")

    assert not service.is_alive(), "libE didn't exit after idle_timeout"
    print("PASS")


if __name__ == "__main__":
    main()
