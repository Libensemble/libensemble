# import persistent_userfunc_support as puf  # SH testing - this may be way to go - sim/gen_support wrapper
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, UNSET_TAG, EVAL_GEN_TAG

info_default = {'persistent': True}

# def sendrecv_mgr_worker_msg(comm, output):
def sendrecv_mgr_worker_msg(comm, output, libE_info=info_default, calc_status=UNSET_TAG, calc_type=EVAL_GEN_TAG):
    """Send message from worker to manager and receive response.

    :param comm: libEnsemble communicator object
    :param output: Output array to be sent to manager
    :returns: message tag, Work dictionary, calc_in array
    """
    # return puf.sendrecv_mgr_worker_msg(comm, output)  # testing
    send_mgr_worker_msg(comm, output, libE_info, calc_status, calc_type)
    return get_mgr_worker_msg(comm)


# def send_mgr_worker_msg(comm, output, ):
def send_mgr_worker_msg(comm, output, libE_info=info_default, calc_status=UNSET_TAG, calc_type=EVAL_GEN_TAG):

    """Send message from worker to manager.

    :param comm: libEnsemble communicator object
    :param output: Output array to be sent to manager
    :returns: None
    """

    if 'comm' in libE_info:
        # Cannot pickle a comm and should not need for return
        libE_info = dict(libE_info)
        libE_info.pop('comm')

    D = {'calc_out': output,
         'libE_info': libE_info,
         'calc_status': calc_status,
         'calc_type': calc_type
         }
    comm.send(calc_type, D)


def get_mgr_worker_msg(comm):
    """Get message to worker from manager.

    :param comm: libEnsemble communicator object
    :returns: message tag, Work dictionary, calc_in array
    """
    tag, Work = comm.recv()
    if tag in [STOP_TAG, PERSIS_STOP]:
        comm.push_to_buffer(tag, Work)
        return tag, Work, None
    data_tag, calc_in = comm.recv()
    # Check for unexpected STOP (e.g. error between sending Work info and rows)
    if data_tag in [STOP_TAG, PERSIS_STOP]:
        comm.push_to_buffer(data_tag, calc_in)
        return data_tag, calc_in, None  # calc_in is signal identifier
    return tag, Work, calc_in
