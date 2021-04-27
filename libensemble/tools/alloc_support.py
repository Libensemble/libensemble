import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG


def avail_worker_ids(W, persistent=None, active_recv=False):
    """Returns available workers (``active == 0``), as an array, filtered by ``persis_state``.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param persistent: Optional Boolean. If specified, also return workers with given persis_state.
    """
    if persistent is None:
        return W['worker_id'][W['active'] == 0]
    if persistent:
        if active_recv:
            return W['worker_id'][np.logical_and(W['persis_state'] != 0,
                                                 W['active_recv'] != 0)]
        else:
            return W['worker_id'][np.logical_and(W['active'] == 0,
                                                 W['persis_state'] != 0)]
    return W['worker_id'][np.logical_and(W['active'] == 0,
                                         W['persis_state'] == 0)]


def count_gens(W):
    """Return the number of active generators in a set of workers.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    """
    return sum(W['active'] == EVAL_GEN_TAG)


def test_any_gen(W):
    """Return True if a generator worker is active.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    """
    return any(W['active'] == EVAL_GEN_TAG)


def count_persis_gens(W):
    """Return the number of active persistent generators in a set of workers.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    """
    return sum(W['persis_state'] == EVAL_GEN_TAG)


def sim_work(Work, i, H_fields, H_rows, persis_info, **libE_info):
    """Add sim work record to given Work array.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param i: Worker ID.
    :param H_fields: Which fields from H to send
    :param persis_info: current persis_info dictionary

    :returns: None
    """
    libE_info['H_rows'] = H_rows
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_SIM_TAG,
               'libE_info': libE_info}


def gen_work(Work, i, H_fields, H_rows, persis_info, **libE_info):
    """Add gen work record to given Work array.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param i: Worker ID.
    :param H_fields: Which fields from H to send
    :param persis_info: current persis_info dictionary

    :returns: None
    """

    # Count total gens
    try:
        gen_work.gen_counter += 1
    except AttributeError:
        gen_work.gen_counter = 1
    libE_info['gen_count'] = gen_work.gen_counter

    libE_info['H_rows'] = H_rows
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_GEN_TAG,
               'libE_info': libE_info}


def all_returned(H, pt_filter=True):
    """Check if all expected points have returned from sim

    :param H: A :doc:`history array<../data_structures/history_array>`
    :param pt_filter: Optional boolean array filtering expected returned points: Default: All True

    :returns: Boolean. True if all expected points have been returned
    """
    # Exclude cancelled points that were not already given out
    excluded_points = H['cancel_requested'] & ~H['given']
    return np.all(H['returned'][pt_filter & ~excluded_points])
