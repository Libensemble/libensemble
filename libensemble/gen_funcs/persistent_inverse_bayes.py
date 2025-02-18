import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "persistent_updater_after_likelihood",
]


def persistent_updater_after_likelihood(H, persis_info, gen_specs, libE_info):
    """ """
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n = len(lb)
    subbatch_size = gen_specs["user"]["subbatch_size"]
    num_subbatches = gen_specs["user"]["num_subbatches"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Receive information from the manager (or a STOP_TAG)
    batch = -1
    tag = None
    w = np.nan
    while tag not in [STOP_TAG, PERSIS_STOP]:
        batch += 1
        H_o = np.zeros(subbatch_size * num_subbatches, dtype=gen_specs["out"])
        if np.all(~np.isnan(w)):
            H_o["weight"] = w
        for j in range(num_subbatches):
            for i in range(subbatch_size):
                row = subbatch_size * j + i
                H_o["x"][row] = persis_info["rand_stream"].uniform(lb, ub, (1, n))
                H_o["subbatch"][row] = j
                H_o["batch"][row] = batch
                H_o["prior"][row] = np.random.randn()
                H_o["prop"][row] = np.random.randn()

        # Send data and get next assignment
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            w = H_o["prior"] + calc_in["like"] - H_o["prop"]

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
