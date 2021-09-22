from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, UNSET_TAG, EVAL_GEN_TAG
import logging
logger = logging.getLogger(__name__)


def sendrecv_mgr_worker_msg(comm, output):
    """Send message from worker to manager and receive response.

    :param comm: libEnsemble communicator object
    :param output: Output array to be sent to manager
    :returns: message tag, Work dictionary, calc_in array
    """
    send_mgr_worker_msg(comm, output)
    return recv_mgr_worker_msg(comm)


def send_mgr_worker_msg(comm, output):
    """Send message from worker to manager.

    :param comm: libEnsemble communicator object
    :param output: Output array to be sent to manager
    :returns: None
    """
    D = {'calc_out': output,
         'libE_info': {'persistent': True},
         'calc_status': UNSET_TAG,
         'calc_type': EVAL_GEN_TAG
         }
    logger.debug('Persistent gen sending data message to manager')
    comm.send(EVAL_GEN_TAG, D)


def recv_mgr_worker_msg(comm):
    """Get message to worker from manager.

    :param comm: libEnsemble communicator object
    :returns: message tag, Work dictionary, calc_in array
    """
    tag, Work = comm.recv()
    if tag in [STOP_TAG, PERSIS_STOP]:
        logger.debug('Persistent gen received signal {} from manager'.format(tag))
        comm.push_to_buffer(tag, Work)
        return tag, Work, None

    logger.debug('Persistent gen received work request from manager')
    data_tag, calc_in = comm.recv()
    # Check for unexpected STOP (e.g. error between sending Work info and rows)
    if data_tag in [STOP_TAG, PERSIS_STOP]:
        logger.debug('Persistent gen received signal {} from manager while expecting work rows'.format(data_tag))
        comm.push_to_buffer(data_tag, calc_in)
        return data_tag, calc_in, None  # calc_in is signal identifier
    logger.debug('Persistent gen received work rows from manager')
    return tag, Work, calc_in
