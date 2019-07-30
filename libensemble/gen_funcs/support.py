from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, UNSET_TAG, EVAL_GEN_TAG
import sys


def sendrecv_mgr_worker_msg(comm, O, status=None):
    """Send message from worker to manager and receive response.
    """
    send_mgr_worker_msg(comm, O)
    return get_mgr_worker_msg(comm, status=status)


def send_mgr_worker_msg(comm, O):
    """Send message from worker to manager.
    """
    assert len(O) == 1
    # print(25*"-", "Sending {}(SimID: {})".format(O[0][0], O['sim_id']), 25*"-",
    #     flush=True)
    D = {'calc_out': O,
         'libE_info': {'persistent': True},
         'calc_status': UNSET_TAG,
         'calc_type': EVAL_GEN_TAG
         }
    comm.send(EVAL_GEN_TAG, D)


def get_mgr_worker_msg(comm, status=None):
    """Get message to worker from manager.
    """
    # print('[Parent]: Expecting a message from Manager...', flush=True)
    tag, Work = comm.recv()
    # print('[Parent]: Received a tag {}'.format(tag), flush=True)
    sys.stdout.flush()
    if tag in [STOP_TAG, PERSIS_STOP]:
        comm.push_back(tag, Work)
        return tag, None, None
    # print('[Parent]: Expecting a message from Manager...', flush=True)
    _, calc_in = comm.recv()
    # print('[Parent]: Received some message.', flush=True)
    return tag, Work, calc_in
