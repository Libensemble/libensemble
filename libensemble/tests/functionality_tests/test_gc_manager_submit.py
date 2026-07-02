"""
Tests libEnsemble's manager-side Globus Compute submission (GC-only mode).

The manager submits simulation work directly to a mocked Globus Compute
endpoint instead of dispatching to local workers. The generator runs on
the manager thread as normal.

Execute via:
   python test_gc_manager_submit.py

No MPI or local workers are needed -- GC-only mode uses local comms with
nworkers acting as the maximum number of concurrent in-flight GC futures.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 1

import concurrent.futures
from unittest import mock

import numpy as np
from gest_api.vocs import VOCS

from libensemble.gen_classes.sampling import UniformSample
from libensemble.libE import libE
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
from libensemble.utils.globus_compute import GCSession


def norm_sim(H, persis_info, sim_specs, libE_info):
    """Evaluate the Euclidean norm of each input point."""
    H_o = np.zeros(len(H), dtype=sim_specs["out"])
    for i in range(len(H)):
        H_o["f"][i] = float(np.linalg.norm(H["x"][i]))
    return H_o, persis_info


def _make_done_future(value):
    f = concurrent.futures.Future()
    f.set_result(value)
    return f


def _make_gc_executor(sim_f):
    executor = mock.MagicMock()
    executor.register_function.return_value = "mock-fid"

    def fake_submit(fid, args):
        result = sim_f(*args)
        return _make_done_future(result)

    executor.submit_to_registered_function.side_effect = fake_submit
    return executor


if __name__ == "__main__":
    GCSession.clear()

    ENDPOINT = "mock-endpoint-uuid"
    SIM_MAX = 20
    N_VIRTUAL_WORKERS = 4

    vocs = VOCS(
        variables={"x0": [-3.0, 3.0], "x1": [-2.0, 2.0]},
        objectives={"f": "MINIMIZE"},
    )

    sim_specs = SimSpecs(
        sim_f=norm_sim,
        inputs=["x"],
        outputs=[("f", float)],
        globus_compute_endpoint=ENDPOINT,
    )

    gen_specs = GenSpecs(
        generator=UniformSample(vocs),
        inputs=["sim_id"],
        persis_in=["f", "sim_id"],
        outputs=[("x", float, (2,))],
        batch_size=N_VIRTUAL_WORKERS,
    )

    libE_specs = LibeSpecs(
        nworkers=N_VIRTUAL_WORKERS,
        comms="local",
        disable_log_files=True,
        safe_mode=False,
    )

    exit_criteria = ExitCriteria(sim_max=SIM_MAX)

    mock_executor = _make_gc_executor(norm_sim)

    with mock.patch.object(GCSession, "_create_executor", return_value=mock_executor):
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)

    assert flag == 0, f"libEnsemble exited with unexpected flag {flag}"
    assert (
        np.sum(H["sim_ended"]) >= SIM_MAX
    ), f"Expected at least {SIM_MAX} completed sims, got {np.sum(H['sim_ended'])}"

    completed = H[H["sim_ended"]]
    assert len(completed) >= SIM_MAX
    assert np.all(completed["f"] >= 0.0), "Unexpected negative norm value"
    assert (
        mock_executor.submit_to_registered_function.call_count >= SIM_MAX
    ), f"Expected at least {SIM_MAX} GC submissions, got {mock_executor.submit_to_registered_function.call_count}"

    print(f"\nGC-only mode: {np.sum(H['sim_ended'])} sims completed via mocked Globus Compute.")
    print(f"Best f value: {completed['f'].min():.6f}")
    print("\nlibEnsemble GC-only functionality test passed.")
