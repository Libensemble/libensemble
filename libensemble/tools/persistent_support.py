from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, UNSET_TAG, EVAL_GEN_TAG, EVAL_SIM_TAG, calc_type_strings
import logging
logger = logging.getLogger(__name__)


class PersistentSupport:
    """A helper class to assist with writing persistent user functions."""

    def __init__(self, libE_info, calc_type):
        """
        Instantiate a new PersistentSupport instance

        :param libE_info: A dictionary containing information about this work request
        :param calc_type: Named integer giving calculation type - EVAL_GEN_TAG or EVAL_SIM_TAG

        """
        self.libE_info = libE_info
        self.comm = self.libE_info['comm']
        self.calc_type = calc_type
        assert self.calc_type in [EVAL_GEN_TAG, EVAL_SIM_TAG], \
            "User function value {} specifies neither a simulator nor generator.".format(self.calc_type)
        self.calc_str = calc_type_strings[self.calc_type]

    def send(self, output, calc_status=UNSET_TAG):
        """
        Send message from worker to manager.

        :param output: Output array to be sent to manager
        :param calc_status: Optional, Provides a task status

        :returns: None

        """
        if 'comm' in self.libE_info:
            # Need to make copy before remove comm as original could be reused
            libE_info = dict(self.libE_info)
            libE_info.pop('comm')
        else:
            libE_info = self.libE_info

        D = {'calc_out': output,
             'libE_info': libE_info,
             'calc_status': calc_status,
             'calc_type': self.calc_type
             }
        logger.debug('Persistent {} function sending data message to manager'.format(self.calc_str))
        self.comm.send(self.calc_type, D)

    def recv(self):
        """
        Receive message to worker from manager.

        :returns: message tag, Work dictionary, calc_in array

        """
        tag, Work = self.comm.recv()  # Receive meta-data or signal
        if tag in [STOP_TAG, PERSIS_STOP]:
            logger.debug('Persistent {} received signal {} from manager'.format(self.calc_str, tag))
            self.comm.push_to_buffer(tag, Work)
            return tag, Work, None
        else:
            logger.debug('Persistent {} received work request from manager'.format(self.calc_str))

        # Update libE_info
        # self.libE_info = Work['libE_info']

        # Only replace rows - keep same dictionary
        self.libE_info['H_rows'] = Work['libE_info']['H_rows']

        data_tag, calc_in = self.comm.recv()  # Receive work rows

        # Check for unexpected STOP (e.g. error between sending Work info and rows)
        if data_tag in [STOP_TAG, PERSIS_STOP]:
            logger.debug('Persistent {} received signal {} '.format(self.calc_str, tag) +
                         'from manager while expecting work rows')
            self.comm.push_to_buffer(data_tag, calc_in)
            return data_tag, calc_in, None  # calc_in is signal identifier

        logger.debug('Persistent {} received work rows from manager'.format(self.calc_str))
        return tag, Work, calc_in

    def send_recv(self, output, calc_status=UNSET_TAG):
        """
        Send message from worker to manager and receive response.

        :param output: Output array to be sent to manager
        :param calc_status: Optional, Provides a task status

        :returns: message tag, Work dictionary, calc_in array

        """
        self.send(output, calc_status)
        return self.recv()
