"""
Wrapper for MOP-MOD
"""
import numpy as np


def mop_mod_wrapper(H, persis_info, gen_specs, _):
    """
    Generates ``gen_specs['gen_batch_size']`` points uniformly over the domain
    defined by ``gen_specs['ub']`` and ``gen_specs['lb']``.

    :See:
        ``libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']

    n = len(lb)
    b = gen_specs['gen_batch_size']

    O = np.zeros(b, dtype=gen_specs['out'])

    O['x'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))

    if len(H) == 0:
        O['iter_num'] = 0
    else:
        O['iter_num'] = np.max(H['iter_num']) + 1

    return O, persis_info
