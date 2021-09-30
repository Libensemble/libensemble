from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, UNSET_TAG, EVAL_GEN_TAG, EVAL_SIM_TAG, calc_type_strings
import logging
logger = logging.getLogger(__name__)

info_default = {'persistent': True}


class PersistentSupport:

    def __init__(self, comm, calc_type):
        self.comm = comm
        self.calc_type = calc_type
        assert self.calc_type in [EVAL_GEN_TAG, EVAL_SIM_TAG], \
            "User function value {} specifies neither a simulator nor generator.".format(self.calc_type)

        self.calc_str = calc_type_strings[self.calc_type]

    def send(self, output, libE_info=info_default, calc_status=UNSET_TAG):
        """Send message from worker to manager.

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
             'calc_type': self.calc_type
             }
        logger.debug('Persistent {} function sending data message to manager'.format(self.calc_str))
        self.comm.send(self.calc_type, D)

    def recv(self):
        """Receive message to worker from manager.

        :returns: message tag, Work dictionary, calc_in array
        """
        tag, Work = self.comm.recv()
        if tag in [STOP_TAG, PERSIS_STOP]:
            logger.debug('Persistent {} received signal {} from manager'.format(self.calc_str, tag))
            if not isinstance(Work, dict):
                self.comm.push_to_buffer(tag, Work)
                return tag, Work, None

        logger.debug('Persistent {} received work request from manager'.format(self.calc_str))
        data_tag, calc_in = self.comm.recv()
        # Check for unexpected STOP (e.g. error between sending Work info and rows)
        if data_tag in [STOP_TAG, PERSIS_STOP]:
            logger.debug('Persistent {} received signal {} '.format(self.calc_str, tag) +
                         'from manager while expecting work rows')
            self.comm.push_to_buffer(data_tag, calc_in)
            return data_tag, calc_in, None  # calc_in is signal identifier

        logger.debug('Persistent {} received work rows from manager'.format(self.calc_str))
        if tag in [STOP_TAG, PERSIS_STOP]:
            self.comm.push_to_buffer(tag, Work)
        return tag, Work, calc_in

    def send_recv(self, output, libE_info=info_default, calc_status=UNSET_TAG):
        """Send message from worker to manager and receive response.

        :param output: Output array to be sent to manager
        :returns: message tag, Work dictionary, calc_in array
        """
        self.send(output, libE_info, calc_status)
        return self.recv()
