from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, UNSET_TAG, EVAL_GEN_TAG


def sendrecv_mgr_worker_msg(comm, output, status=None):
    """Send message from worker to manager and receive response.
    """
    send_mgr_worker_msg(comm, output)
    return get_mgr_worker_msg(comm, status=status)


def send_mgr_worker_msg(comm, output):
    """Send message from worker to manager.
    """
    D = {'calc_out': output,
         'libE_info': {'persistent': True},
         'calc_status': UNSET_TAG,
         'calc_type': EVAL_GEN_TAG
         }
    comm.send(EVAL_GEN_TAG, D)


def get_mgr_worker_msg(comm, status=None):
    """Get message to worker from manager.
    """
    tag, Work = comm.recv()
    if tag in [STOP_TAG, PERSIS_STOP]:
        comm.push_to_buffer(tag, Work)
        return tag, Work, None
    _, calc_in = comm.recv()
    return tag, Work, calc_in
