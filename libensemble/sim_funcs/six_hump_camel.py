"""
This module contains various versions that evaluate the six-hump camel function.

Six-hump camel function is documented here:
  https://www.sfu.ca/~ssurjano/camel6.html

"""
__all__ = [
    "six_hump_camel",
    "six_hump_camel_simple",
    "persistent_six_hump_camel",
]

import sys
import time
import numpy as np

from libensemble.message_numbers import (
    EVAL_SIM_TAG,
    FINISHED_PERSISTENT_SIM_TAG,
    PERSIS_STOP,
    STOP_TAG,
)
from libensemble.tools.persistent_support import PersistentSupport


def six_hump_camel(H, persis_info, sim_specs, libE_info):
    """
    Evaluates the six hump camel function for a collection of points given in ``H["x"]``.
    Additionally evaluates the gradient if ``"grad"`` is a field in
    ``sim_specs["out"]`` and pauses for ``sim_specs["user"]["pause_time"]]`` if
    defined.

    .. seealso::
        `test_old_aposmm_with_gradients.py  <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_old_aposmm_with_gradients.py>`_ # noqa
    """

    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        H_o["f"][i] = six_hump_camel_func(x)

        if "grad" in H_o.dtype.names:
            H_o["grad"][i] = six_hump_camel_grad(x)

        if "user" in sim_specs and "pause_time" in sim_specs["user"]:
            time.sleep(sim_specs["user"]["pause_time"])

    return H_o, persis_info


def six_hump_camel_simple(x, _, sim_specs):
    """
    Evaluates the six hump camel function for a single point ``x``.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    H_o = np.zeros(1, dtype=sim_specs["out"])

    H_o["f"] = six_hump_camel_func(x[0][0])

    if "pause_time" in sim_specs["user"]:
        time.sleep(sim_specs["user"]["pause_time"])

    return H_o


def persistent_six_hump_camel(H, persis_info, sim_specs, libE_info):
    """
    Similar to ``six_hump_camel``, but runs in persistent mode.
    """

    ps = PersistentSupport(libE_info, EVAL_SIM_TAG)

    # Either start with a work item to process - or just start and wait for data
    if H.size > 0:
        tag = None
        Work = None
        calc_in = H
    else:
        tag, Work, calc_in = ps.recv()

    while tag not in [STOP_TAG, PERSIS_STOP]:
        # calc_in: This should either be a function (unpack_work ?) or included/unpacked in ps.recv/ps.send_recv.
        if Work is not None:
            persis_info = Work.get("persis_info", persis_info)
            libE_info = Work.get("libE_info", libE_info)

        # Call standard six_hump_camel sim
        H_o, persis_info = six_hump_camel(calc_in, persis_info, sim_specs, libE_info)

        tag, Work, calc_in = ps.send_recv(H_o)

    final_return = None

    # Overwrite final point - for testing only
    if sim_specs["user"].get("replace_final_fields", 0):
        calc_in = np.ones(1, dtype=[("x", float, (2,))])
        H_o, persis_info = six_hump_camel(calc_in, persis_info, sim_specs, libE_info)
        final_return = H_o

    return final_return, persis_info, FINISHED_PERSISTENT_SIM_TAG


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    return term1 + term2 + term3


def six_hump_camel_grad(x):
    """
    Definition of the six-hump camel gradient
    """

    x1 = x[0]
    x2 = x[1]
    grad = np.zeros(2)

    grad[0] = 2.0 * (x1**5 - 4.2 * x1**3 + 4.0 * x1 + 0.5 * x2)
    grad[1] = x1 + 16 * x2**3 - 8 * x2

    return grad


if __name__ == "__main__":
    x = (float(sys.argv[1]), float(sys.argv[2]))
    result = six_hump_camel_func(x)
    print(result)
