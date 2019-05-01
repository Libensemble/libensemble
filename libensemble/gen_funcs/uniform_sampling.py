"""
This module contains multiple generation functions for sampling a domain. All
use (and return) a random stream in ``persis_info``, given by the allocation
function.
"""
import numpy as np


def uniform_random_sample_with_different_nodes_and_ranks(H, persis_info, gen_specs, _):
    """
    Generates points uniformly over the domain defined by ``gen_specs['ub']`` and
    ``gen_specs['lb']``. Also randomly requests a different ``number_of_nodes``
    and ``ranks_per_node`` to be used in the evaluation of the generated point.

    :See:
        ``libensemble/tests/regression_tests/test_6-hump_camel_with_different_nodes_uniform_sample.py``
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']
    n = len(lb)

    if len(H) == 0:
        b = gen_specs['initial_batch_size']

        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0, b):
            x = persis_info['rand_stream'].uniform(lb, ub, (1, n))
            O['x'][i] = x
            O['num_nodes'][i] = 1
            O['ranks_per_node'][i] = 16
            O['priority'] = 1

    else:
        O = np.zeros(1, dtype=gen_specs['out'])
        O['x'] = len(H)*np.ones(n)
        O['num_nodes'] = np.random.randint(1, gen_specs['max_num_nodes']+1)
        O['ranks_per_node'] = np.random.randint(1, gen_specs['max_ranks_per_node']+1)
        O['priority'] = 10*O['num_nodes']

    return O, persis_info


def uniform_random_sample_obj_components(H, persis_info, gen_specs, _):
    """
    Generates points uniformly over the domain defined by ``gen_specs['ub']``
    and ``gen_specs['lb']`` but requests each ``obj_component`` be evaluated
    separately.

    :See:
        ``libensemble/tests/regression_tests/test_chwirut_uniform_sampling_one_residual_at_a_time.py``
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']

    n = len(lb)
    m = gen_specs['components']
    b = gen_specs['gen_batch_size']

    O = np.zeros(b*m, dtype=gen_specs['out'])
    for i in range(0, b):
        x = persis_info['rand_stream'].uniform(lb, ub, (1, n))

        O['x'][i*m:(i+1)*m, :] = np.tile(x, (m, 1))
        O['priority'][i*m:(i+1)*m] = persis_info['rand_stream'].uniform(0, 1, m)
        O['obj_component'][i*m:(i+1)*m] = np.arange(0, m)

        O['pt_id'][i*m:(i+1)*m] = len(H)//m+i

    return O, persis_info


def uniform_random_sample(H, persis_info, gen_specs, _):
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

    return O, persis_info
