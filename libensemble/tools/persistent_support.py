import logging

import numpy as np
import numpy.typing as npt

from libensemble.message_numbers import EVAL_GEN_TAG, EVAL_SIM_TAG, PERSIS_STOP, STOP_TAG, UNSET_TAG, calc_type_strings

logger = logging.getLogger(__name__)


class PersistentSupport:
    """A helper class to assist with writing persistent user functions."""

    def __init__(self, libE_info: dict[str, dict], calc_type: int) -> None:
        """
        Instantiate a new PersistentSupport instance

        :param libE_info: A dictionary containing information about this work request
        :param calc_type: Named integer giving calculation type - EVAL_GEN_TAG or EVAL_SIM_TAG

        """
        self.libE_info = libE_info
        self.comm = self.libE_info["comm"]
        self.calc_type = calc_type
        assert self.calc_type in [
            EVAL_GEN_TAG,
            EVAL_SIM_TAG,
        ], f"The calc_type: {self.calc_type} specifies neither a simulator nor generator."
        self.calc_str = calc_type_strings[self.calc_type]

    def send(self, output: npt.NDArray, calc_status: int = UNSET_TAG, keep_state=False) -> None:
        """
        Send message from worker to manager.

        :param output: Output array to be sent to manager.
        :param calc_status: (Optional) Provides a task status.
        :param keep_state: (Optional) If True the manager will not modify its
            record of the workers state (usually the manager changes the
            worker's state to inactive, indicating the worker is ready to receive
            more work, unless using active receive mode).

        """
        if "comm" in self.libE_info:
            # Need to make copy before remove comm as original could be reused
            libE_info = dict(self.libE_info)
            libE_info.pop("comm")
            libE_info.pop("executor")
        else:
            libE_info = self.libE_info

        libE_info["keep_state"] = keep_state

        D = {
            "calc_out": output,
            "libE_info": libE_info,
            "calc_status": calc_status,
            "calc_type": self.calc_type,
        }
        logger.debug(f"Persistent {self.calc_str} function sending data message to manager")
        self.comm.send(self.calc_type, D)

    def recv(self, blocking: bool = True) -> (int, dict, npt.NDArray):
        """
        Receive message to worker from manager.

        :param blocking: (Optional) If True (default), will block until a message is received.

        :returns: message tag, Work dictionary, calc_in array

        """

        if not blocking:
            if not self.comm.mail_flag():
                return None, None, None

        tag, Work = self.comm.recv()  # Receive meta-data or signal
        if tag in [STOP_TAG, PERSIS_STOP]:
            logger.debug(f"Persistent {self.calc_str} received signal {tag} from manager")
            if not isinstance(Work, dict):
                self.comm.push_to_buffer(tag, Work)
                return tag, Work, None
        else:
            logger.debug(f"Persistent {self.calc_str} received work request from manager")

        # Update libE_info
        # self.libE_info = Work['libE_info']

        # Only replace rows - keep same dictionary
        self.libE_info["H_rows"] = Work["libE_info"]["H_rows"]

        data_tag, calc_in = self.comm.recv()  # Receive work rows

        # Check for unexpected STOP (e.g., error between sending Work info and rows)
        if data_tag in [STOP_TAG, PERSIS_STOP]:
            logger.debug(
                f"Persistent {self.calc_str} received signal {tag} " + "from manager while expecting work rows"
            )
            self.comm.push_to_buffer(data_tag, calc_in)
            return data_tag, calc_in, None  # calc_in is signal identifier

        logger.debug(f"Persistent {self.calc_str} received work rows from manager")
        return tag, Work, calc_in

    def send_recv(self, output: npt.NDArray, calc_status: int = UNSET_TAG) -> (int, dict, npt.NDArray):
        """
        Send message from worker to manager and receive response.

        :param output: Output array to be sent to manager.
        :param calc_status: (Optional) Provides a task status.

        :returns: message tag, Work dictionary, calc_in array

        """
        self.send(output, calc_status)
        return self.recv()

    def request_cancel_sim_ids(self, sim_ids: list[int]):
        """Request cancellation of sim_ids.

        :param sim_ids: A list of sim_ids to cancel.

        A message is sent to the manager to mark requested sim_ids as cancel_requested.
        """
        H_o = np.zeros(len(sim_ids), dtype=[("sim_id", int), ("cancel_requested", bool)])
        H_o["sim_id"] = sim_ids
        H_o["cancel_requested"] = True
        self.send(H_o, keep_state=True)
