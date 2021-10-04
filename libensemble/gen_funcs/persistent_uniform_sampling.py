import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = ['persistent_uniform',
           'uniform_random_sample_with_variable_resources',
           'persistent_request_shutdown']


def persistent_uniform(H, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs['initial_batch_size']`` uniformly sampled points the first time it
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
    b = gen_specs['user']['initial_batch_size']
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs['out'])
        H_o['x'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, '__len__'):
            b = len(calc_in)

    H_o = None
    if gen_specs['user'].get('replace_final_fields', 0):
        # This is only to test libE ability to accept History after a
        # PERSIS_STOP. This history is returned in Work.
        H_o = Work
        H_o['x'] = -1.23

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def uniform_random_sample_with_variable_resources(H, persis_info, gen_specs, libE_info):
    """
    Generates points uniformly over the domain defined by ``gen_specs['user']['ub']`` and
    ``gen_specs['user']['lb']``. Also randomly requests a different number of resource
    sets to be used in the evaluation of the generated points after the initial batch.

    .. seealso::
        `test_uniform_sampling_with_variable_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_with_variable_resources.py>`_ # noqa
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(lb)
    b = gen_specs['user']['initial_batch_size']
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    H_o = np.zeros(b, dtype=gen_specs['out'])
    for i in range(0, b):
        # x= i*np.ones(n)
        x = persis_info['rand_stream'].uniform(lb, ub, (1, n))
        H_o['x'][i] = x
        H_o['resource_sets'][i] = 1
        H_o['priority'] = 1

    # Send batches until manager sends stop tag
    tag, Work, calc_in = ps.send_recv(H_o)
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs['out'])
        # H_o['x'] = len(H)*np.ones(n)
        H_o['x'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))
        H_o['resource_sets'] = persis_info['rand_stream'].randint(1, gen_specs['user']['max_resource_sets']+1, b)
        H_o['priority'] = 10*H_o['resource_sets']
        print('Created {} sims, with worker_teams req. of size(s) {}'.format(b, H_o['resource_sets']), flush=True)
        tag, Work, calc_in = ps.send_recv(H_o)

        if calc_in is not None:
            b = len(calc_in)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_request_shutdown(H, persis_info, gen_specs, libE_info):
    """
    This generation function is similar in structure to persistent_uniform,
    but uses a count to test exiting on a threshold value. This principle can
    be used with a supporting allocation function (e.g. start_only_persistent)
    to shutdown an ensemble when a condition is met.

    .. seealso::
        `test_persistent_uniform_gen_decides_stop.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_gen_decides_stop.py>`_ # noqa
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(lb)
    b = gen_specs['user']['initial_batch_size']
    shutdown_limit = gen_specs['user']['shutdown_limit']
    f_count = 0
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = np.zeros(b, dtype=gen_specs['out'])
        H_o['x'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))
        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, '__len__'):
            b = len(calc_in)
        f_count += b
        if f_count >= shutdown_limit:
            print('Reached threshold.', f_count, flush=True)
            break  # End the persistent gen

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
