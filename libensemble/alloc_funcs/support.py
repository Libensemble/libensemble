import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG


def avail_worker_ids(W, persistent=None):
    "Get available workers (active == 0), filtered by persis_state."
    if persistent is None:
        return W['worker_id'][W['active'] == 0]
    if persistent:
        return W['worker_id'][np.logical_and(W['active'] == 0,
                                             W['persis_state'] != 0)]
    return W['worker_id'][np.logical_and(W['active'] == 0,
                                         W['persis_state'] == 0)]


def count_gens(W):
    "Return the number of generators in a set of workers."
    return sum(W['active'] == EVAL_GEN_TAG)


def count_persis_gens(W):
    "Return the number of persistent generators in a set of workers."
    return sum(W['persis_state'] == EVAL_GEN_TAG)


def sim_work(Work, i, H_fields, H_rows, persis_info, **libE_info):
    "Add sim work record to work array."
    libE_info['H_rows'] = H_rows
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_SIM_TAG,
               'libE_info': libE_info}


def gen_work(Work, i, H_fields, H_rows, persis_info, **libE_info):
    "Add gen work record to work array."
    libE_info['H_rows'] = H_rows
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_GEN_TAG,
               'libE_info': libE_info}
