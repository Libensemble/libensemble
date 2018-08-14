from __future__ import division
from __future__ import absolute_import
import numpy as np

from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.alloc_funcs.support import avail_worker_ids


def give_sim_work_first(W, H, sim_specs, gen_specs, persis_info):
    """
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most ``gen_specs['num_inst']``)
    generator instances.

    Allows for a ``'batch_mode'`` where no generation
    work is given out unless all entries in ``H`` are either returned or
    paused.

    Allows for ``blocking`` of workers that are not active, for example, so
    their resources can be used for a different simulation evaluation.

    Can give points in highest priority, if ``'priority'`` is a field in ``H``.

    This is the default allocation function if one is not defined.

    :See:
        ``/libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``
    """

    Work = {}
    gen_count = sum(W['active'] == EVAL_GEN_TAG)
    blocked_set = set(W['worker_id'][W['blocked']])

    for i in avail_worker_ids(W):

        # Only consider giving to worker i if it's resources are not blocked by some other calculation
        if i in blocked_set:
            continue

        # Find indices of H that are not given nor paused
        if not np.all(H['allocated']):
            # Give sim work if possible

            # Pick all high priority, oldest high priority, or just oldest point
            if 'priority' in H.dtype.fields:
                priorities = H['priority'][~H['allocated']]
                if gen_specs.get('give_all_with_same_priority'):
                    q_inds = (priorities == np.max(priorities))
                else:
                    q_inds = np.argmax(priorities)
            else:
                q_inds = 0

            sim_ids_to_send = np.nonzero(~H['allocated'])[0][q_inds]
            sim_ids_to_send = np.atleast_1d(sim_ids_to_send)

            # Only give work if enough idle workers
            if 'num_nodes' in H.dtype.names and np.any(H[sim_ids_to_send]['num_nodes'] > 1):
                if np.any(H[sim_ids_to_send]['num_nodes'] > sum(W['active'] == 0) - len(Work) - len(blocked_set)):
                    # Worker i doesn't get any work. Just waiting for other resources to open up
                    continue
                block_others = True
            else:
                block_others = False

            Work[i] = {'H_fields': sim_specs['in'],
                       'persis_info': {},
                       'tag': EVAL_SIM_TAG,
                       'libE_info': {'H_rows': sim_ids_to_send},
                      }
            H['allocated'][sim_ids_to_send] = True

            if block_others:
                unassigned_workers = set(avail_worker_ids(W)) - set(Work.keys()) - blocked_set
                workers_to_block = list(unassigned_workers)[:np.max(H[sim_ids_to_send]['num_nodes'])-1]
                blocked_set.update(workers_to_block)
                Work[i]['libE_info']['blocking'] = workers_to_block

        else:

            # Allow at most num_inst active generator instances
            if gen_count >= gen_specs.get('num_inst', gen_count+1):
                break

            # Don't give out gen instances in batch mode if workers still working
            still_working = np.logical_and(~H['returned'], ~H['paused'])
            if gen_specs.get('batch_mode') and np.any(still_working):
                break

            # Give gen work
            gen_count += 1
            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': gen_specs['in'],
                       'tag': EVAL_GEN_TAG,
                       'libE_info': {'H_rows': range(len(H))}
                      }

    return Work, persis_info
