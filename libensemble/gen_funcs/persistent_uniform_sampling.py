import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg


def persistent_uniform(H, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs['gen_batch_size']`` uniformly sampled points the first time it
    is called. Afterwards, it returns the number of points given. This can be
    used in either a batch or asynchronous mode by adjusting the allocation
    function.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(lb)
    b = gen_specs['user']['gen_batch_size']

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs['out'])
        H_o['x'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            b = len(calc_in)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


# SH TODO: Change name of function now
#          and check/update docstring
def uniform_random_sample_with_different_resources(H, persis_info, gen_specs, libE_info):
    """
    Generates points uniformly over the domain defined by ``gen_specs['user']['ub']`` and
    ``gen_specs['user']['lb']``. Also randomly requests a different ``number_of_nodes``
    and ``ranks_per_node`` to be used in the evaluation of the generated point.

    .. seealso::
        `test_uniform_sampling_with_different_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_with_different_resources.py>`_ # noqa
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(lb)
    b = gen_specs['user']['initial_batch_size']

    H_o = np.zeros(b, dtype=gen_specs['out'])

    # SH TODO: Can probably do like function above, without loop
    for i in range(0, b):
        # x= i*np.ones(n)
        x = persis_info['rand_stream'].uniform(lb, ub, (1, n))
        H_o['x'][i] = x
        H_o['resource_sets'][i] = 1
        H_o['priority'] = 1

    # Send batches until manager sends stop tag
    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs['out'])

        # SH TODO: Re-evaluate - x values - maybe should be clear this is only for a test...
        # H_o['x'] = len(H)*np.ones(n)
        H_o['x'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))

        # SH TODO: Should we use persis_info['rand_stream'] for this also?
        H_o['resource_sets'] = np.random.randint(1, gen_specs['user']['max_resource_sets']+1, b)
        H_o['priority'] = 10*H_o['resource_sets']
        # print('Created {} sims, with worker_teams req. of size(s) {}'.format(b, H_o['resource_sets']), flush=True)
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        if calc_in is not None:
            b = len(calc_in)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
