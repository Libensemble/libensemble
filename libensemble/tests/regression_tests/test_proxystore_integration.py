"""
Test libEnsemble's capability to use no gen_f and instead coordinates the
evaluation of an existing set of points.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_evaluate_existing_sample.py
   python test_evaluate_existing_sample.py --nworkers 3
   python test_evaluate_existing_sample.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_OS_SKIP: OSX WIN
# TESTSUITE_EXTRA: true

from pathlib import Path

import numpy as np

# Import libEnsemble items for this test
from libensemble import Ensemble
from libensemble.alloc_funcs.give_pregenerated_work import give_pregenerated_sim_work as alloc_f
from libensemble.sim_funcs.borehole import gen_borehole_input
from libensemble.specs import AllocSpecs, ExitCriteria, SimSpecs, input_fields, output_data


def insert_proxy(H0):
    from proxystore.connectors.redis import RedisConnector
    from proxystore.store import Store, get_store

    store = Store(
        "my-store",
        RedisConnector(hostname="localhost", port=6379),
        register=True,
    )

    store = get_store("my-store")
    picture = Path("libE_logo.png").absolute().read_bytes()
    proxy = store.proxy(picture)
    for i in range(len(H0)):
        H0[i]["proxy"] = proxy


def check_H(H):
    from proxystore.proxy import Proxy

    assert all([isinstance(H[i]["proxy"], Proxy) for i in range(len(H))])


@input_fields(["x", "proxy"])
@output_data([("f", float)])
def one_d_example(x, persis_info, sim_specs, info):

    H_o = np.zeros(1, dtype=sim_specs["out"])
    H_o["f"] = np.linalg.norm(x["x"])
    picture = bytes(x["proxy"][0])

    sim_id = info["H_rows"][0]
    worker_id = info["workerID"]

    Path(f"logo_id{sim_id}_worker{worker_id}.png").write_bytes(picture)

    return H_o, persis_info


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    n_samp = 4
    H0 = np.zeros(n_samp, dtype=[("x", float, 8), ("sim_id", int), ("sim_started", bool), ("proxy", object)])
    np.random.seed(0)
    H0["x"] = gen_borehole_input(n_samp)
    H0["sim_id"] = range(n_samp)
    H0["sim_started"] = False
    insert_proxy(H0)

    sampling = Ensemble(parse_args=True)
    sampling.H0 = H0
    sampling.sim_specs = SimSpecs(sim_f=one_d_example)
    sampling.alloc_specs = AllocSpecs(alloc_f=alloc_f)
    sampling.exit_criteria = ExitCriteria(sim_max=len(H0))
    sampling.run()

    if sampling.is_manager:
        assert len(sampling.H) == len(H0)
        assert np.array_equal(H0["x"], sampling.H["x"])
        assert np.all(sampling.H["sim_ended"])
        check_H(sampling.H)
        print("\nlibEnsemble correctly didn't add anything to initial sample")
        sampling.save_output(__file__)
