"""Test for QueueGenerator

Spins up libE in a thread with the QueueGenerator + a trivial doubler sim,
submits N work items, drains results, sends shutdown sentinel, joins.

Run with:
    python test_queue_gen.py
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4

import threading
import time
from queue import Empty, Queue

import numpy as np
from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes.queue_gen import QueueGenerator
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

NWORKERS = 4
N_SUBMITS = 10


def doubler_sim(InputArray, _, sim_specs):
    """Trivial sim: returns 2*x. Sleeps a bit to mimic real work."""
    time.sleep(0.5)
    out = np.zeros(1, dtype=sim_specs["out"])
    out["y"] = 2.0 * InputArray["x"][0]
    return out


def run_libe(input_q, output_q, nworkers, sim_max):
    vocs = VOCS(variables={"x": [-100.0, 100.0]}, objectives={"y": "MINIMIZE"})
    gen = QueueGenerator(vocs, input_queue=input_q, output_queue=output_q)

    # service_mode=True lets the manager tolerate idle workers + empty alloc
    # while we wait for new submissions on the queue.
    libE_specs = LibeSpecs(
        nworkers=nworkers,
        service_mode=True,
        final_gen_send=True,  # ensure last batch of completed sims is ingested before stop
    )
    sim_specs = SimSpecs(sim_f=doubler_sim, inputs=["x"], outputs=[("y", float)])
    gen_specs = GenSpecs(generator=gen, vocs=vocs, persis_in=["x", "y"], batch_size=1)
    # QueueGenerator.shutdown_sentinel() which makes suggest return [] forever.
    exit_criteria = ExitCriteria(sim_max=sim_max)

    workflow = Ensemble(sim_specs, gen_specs, exit_criteria, libE_specs)
    workflow.run()


def main():
    input_q: Queue = Queue()
    output_q: Queue = Queue()

    libe_thread = threading.Thread(
        target=run_libe, args=(input_q, output_q, NWORKERS, N_SUBMITS), daemon=True
    )
    libe_thread.start()
    print(f"libE thread started ({NWORKERS} workers)")

    # Submit work (1, 2, 3....), then  signal shutdown.
    for i in range(N_SUBMITS):
        input_q.put({"x": float(i)})
    input_q.put(QueueGenerator.shutdown_sentinel())
    print(f"submitted {N_SUBMITS} items + shutdown sentinel")

    # Drain until we have all results (or timeout)
    results = []
    deadline = time.time() + 60
    while len(results) < N_SUBMITS and time.time() < deadline:
        try:
            results.append(output_q.get(timeout=1))
            print(f"  got: {results[-1]}")
        except Empty:
            pass

    print(f"\ncollected {len(results)}/{N_SUBMITS}")

    # Verify y == 2*x
    ok = True
    for r in results:
        x = r.get("x")
        y = r.get("y")
        if x is None or y is None or abs(y - 2 * x) > 1e-9:
            print(f"  MISMATCH: {r}")
            ok = False
    print("PASS" if ok else "FAIL")

    # Wait for libE to wind down
    libe_thread.join(timeout=10)
    print("done")


if __name__ == "__main__":
    main()
