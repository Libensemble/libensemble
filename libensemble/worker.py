"""
libEnsemble worker class
====================================================
"""

import socket
import logging
import logging.handlers
from itertools import count
from traceback import format_exc
from traceback import format_exception_only as format_exc_msg

import numpy as np

from libensemble.message_numbers import \
    EVAL_SIM_TAG, EVAL_GEN_TAG, \
    UNSET_TAG, STOP_TAG, PERSIS_STOP, CALC_EXCEPTION
from libensemble.message_numbers import MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL
from libensemble.message_numbers import calc_type_strings, calc_status_strings
from libensemble.output_directory import EnsembleDirectory

from libensemble.utils.timer import Timer
from libensemble.executors.executor import Executor
from libensemble.resources.resources import Resources
from libensemble.comms.logs import worker_logging_config
from libensemble.comms.logs import LogConfig
import cProfile
import pstats

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)
task_timing = False


def worker_main(comm, sim_specs, gen_specs, libE_specs, workerID=None, log_comm=True):
    """Evaluates calculations given to it by the manager.

    Creates a worker object, receives work from manager, runs worker,
    and communicates results. This routine also creates and writes to
    the workers summary file.

    Parameters
    ----------
    comm: communicator
        Comm object for manager communications

    sim_specs: dict
        Parameters/information for simulation calculations

    gen_specs: dict
        Parameters/information for generation calculations

    libE_specs: dict
        Parameters/information for libE operations

    workerID: int
        Manager assigned worker ID (if None, default is comm.rank)

    log_comm: boolean
        Whether to send logging over comm
    """

    if libE_specs.get('profile'):
        pr = cProfile.Profile()
        pr.enable()

    # Receive dtypes from manager
    _, dtypes = comm.recv()
    workerID = workerID or comm.rank

    # Initialize logging on comms
    if log_comm:
        worker_logging_config(comm, workerID)

    # Set up and run worker
    worker = Worker(comm, dtypes, workerID, sim_specs, gen_specs, libE_specs)
    worker.run()

    if libE_specs.get('profile'):
        pr.disable()
        profile_state_fname = 'worker_%d.prof' % (workerID)

        with open(profile_state_fname, 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_stats()


######################################################################
# Worker Class
######################################################################


class WorkerErrMsg:

    def __init__(self, msg, exc):
        self.msg = msg
        self.exc = exc


class Worker:

    """The worker class provides methods for controlling sim and gen funcs

    **Object Attributes:**

    These are public object attributes.

    :ivar comm communicator:
        Comm object for manager communications

    :ivar dict dtypes:
        Dictionary containing type information for sim and gen inputs

    :ivar int workerID:
        The libensemble Worker ID

    :ivar dict sim_specs:
        Parameters/information for simulation calculations

    :ivar dict calc_iter:
        Dictionary containing counts for each type of calc (e.g. sim or gen)
    """

    def __init__(self, comm, dtypes, workerID, sim_specs, gen_specs, libE_specs):
        """Initializes new worker object

        """
        self.comm = comm
        self.dtypes = dtypes
        self.workerID = workerID
        self.sim_specs = sim_specs
        self.libE_specs = libE_specs
        self.calc_iter = {EVAL_SIM_TAG: 0, EVAL_GEN_TAG: 0}
        self._run_calc = Worker._make_runners(sim_specs, gen_specs)
        # self._calc_id_counter = count()
        Worker._set_executor(self.workerID, self.comm)
        Worker._set_resources(self.workerID, self.comm)
        self.EnsembleDirectory = EnsembleDirectory(libE_specs=libE_specs)

    @staticmethod
    def _make_runners(sim_specs, gen_specs):
        """Creates functions to run a sim or gen"""

        sim_f = sim_specs['sim_f']

        def run_sim(calc_in, persis_info, libE_info):
            """Calls the sim func."""
            return sim_f(calc_in, persis_info, sim_specs, libE_info)

        if gen_specs:
            gen_f = gen_specs['gen_f']

            def run_gen(calc_in, persis_info, libE_info):
                """Calls the gen func."""
                return gen_f(calc_in, persis_info, gen_specs, libE_info)
        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    @staticmethod
    def _set_rset_team(rset_team):
        """Pass new rset_team to worker resources"""
        resources = Resources.resources
        if isinstance(resources, Resources):
            resources.worker_resources.set_rset_team(rset_team)
            return True
        else:
            return False

    @staticmethod
    def _set_executor(workerID, comm):
        """Sets worker ID in the executor, return True if set"""
        exctr = Executor.executor
        if isinstance(exctr, Executor):
            exctr.set_worker_info(comm, workerID)  # When merge update
            return True
        else:
            logger.debug("No xecutor set on worker {}".format(workerID))
            return False

    @staticmethod
    def _set_resources(workerID, comm):
        """Sets worker ID in the resources, return True if set"""
        resources = Resources.resources
        if isinstance(resources, Resources):
            resources.set_worker_resources(comm.get_num_workers(), workerID)
            return True
        else:
            logger.debug("No resources set on worker {}".format(workerID))
            return False

    def _handle_calc(self, Work, calc_in):
        """Runs a calculation on this worker object.

        This routine calls the user calculations. Exceptions are caught,
        dumped to the summary file, and raised.

        Parameters
        ----------

        Work: :obj:`dict`
            :ref:`(example)<datastruct-work-dict>`

        calc_in: obj: numpy structured array
            Rows from the :ref:`history array<datastruct-history-array>`
            for processing
        """
        calc_type = Work['tag']
        self.calc_iter[calc_type] += 1

        # calc_stats stores timing and summary info for this Calc (sim or gen)
        # calc_id = next(self._calc_id_counter)

        # from output_directory.py
        if calc_type == EVAL_SIM_TAG:
            enum_desc = 'sim_id'
            calc_id = EnsembleDirectory.extract_H_ranges(Work)
        else:
            enum_desc = 'Gen no'
            # Use global gen count if available
            if Work['libE_info'].get('gen_count'):
                calc_id = str(Work['libE_info']['gen_count'])
            else:
                calc_id = str(self.calc_iter[calc_type])
        # Add a right adjust (minimum width).
        calc_id = calc_id.rjust(5, ' ')

        timer = Timer()

        try:
            logger.debug("Starting {}: {}".format(enum_desc, calc_id))
            calc = self._run_calc[calc_type]
            with timer:
                if self.EnsembleDirectory.use_calc_dirs(calc_type):
                    loc_stack, calc_dir = self.EnsembleDirectory.prep_calc_dir(Work, self.calc_iter,
                                                                               self.workerID, calc_type)
                    with loc_stack.loc(calc_dir):  # Changes to calculation directory
                        out = calc(calc_in, Work['persis_info'], Work['libE_info'])
                else:
                    out = calc(calc_in, Work['persis_info'], Work['libE_info'])

                logger.debug("Returned from user function for {} {}".format(enum_desc, calc_id))

            assert isinstance(out, tuple), \
                "Calculation output must be a tuple."
            assert len(out) >= 2, \
                "Calculation output must be at least two elements."

            calc_status = out[2] if len(out) >= 3 else UNSET_TAG

            if calc_status is None:
                calc_status = UNSET_TAG

            # Check for buffered receive
            if self.comm.recv_buffer:
                tag, message = self.comm.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    if message is MAN_SIGNAL_FINISH:
                        calc_status = MAN_SIGNAL_FINISH

            return out[0], out[1], calc_status
        except Exception as e:
            logger.debug("Re-raising exception from calc {}".format(e))
            calc_status = CALC_EXCEPTION
            raise
        finally:
            # This was meant to be handled by calc_stats module.
            if task_timing and Executor.executor.list_of_tasks:
                # Initially supporting one per calc. One line output.
                task = Executor.executor.list_of_tasks[-1]
                calc_msg = "{} {}: {} {} {} Status: {}".\
                    format(enum_desc,
                           calc_id,
                           calc_type_strings[calc_type],
                           timer,
                           task.timer,
                           calc_status_strings.get(calc_status, "Not set"))
            else:
                calc_msg = "{} {}: {} {} Status: {}".\
                    format(enum_desc,
                           calc_id,
                           calc_type_strings[calc_type],
                           timer,
                           calc_status_strings.get(calc_status, "Not set"))

            logging.getLogger(LogConfig.config.stats_name).info(calc_msg)

    def _recv_H_rows(self, Work):
        """Unpacks Work request and receives any history rows"""

        libE_info = Work['libE_info']
        calc_type = Work['tag']
        if len(libE_info['H_rows']) > 0:
            _, calc_in = self.comm.recv()
        else:
            calc_in = np.zeros(0, dtype=self.dtypes[calc_type])

        logger.debug("Received calc_in ({}) of len {}".
                     format(calc_type_strings[calc_type], np.size(calc_in)))
        assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], \
            "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"

        return libE_info, calc_type, calc_in

    def _handle(self, Work):
        """Handles a work request from the manager"""

        # Check work request and receive second message (if needed)
        libE_info, calc_type, calc_in = self._recv_H_rows(Work)

        # Call user function
        libE_info['comm'] = self.comm
        libE_info['workerID'] = self.workerID
        libE_info['rset_team'] = libE_info.get('rset_team', [])
        Worker._set_rset_team(libE_info['rset_team'])

        calc_out, persis_info, calc_status = self._handle_calc(Work, calc_in)

        if 'libE_info' in Work:
            libE_info = Work['libE_info']

        if 'comm' in libE_info:
            del libE_info['comm']

        # If there was a finish signal, bail
        if calc_status == MAN_SIGNAL_FINISH:
            return None

        # Otherwise, send a calc result back to manager
        logger.debug("Sending to Manager with status {}".format(calc_status))
        return {'calc_out': calc_out,
                'persis_info': persis_info,
                'libE_info': libE_info,
                'calc_status': calc_status,
                'calc_type': calc_type}

    def run(self):
        """Runs the main worker loop."""

        try:
            logger.info("Worker {} initiated on node {}".
                        format(self.workerID, socket.gethostname()))

            for worker_iter in count(start=1):
                logger.debug("Iteration {}".format(worker_iter))

                mtag, Work = self.comm.recv()

                if mtag == STOP_TAG:
                    if Work is MAN_SIGNAL_FINISH:
                        break
                    elif Work is MAN_SIGNAL_KILL:
                        continue

                # Active recv is for persistent worker only - throw away here
                if Work.get('libE_info', False):
                    if Work['libE_info'].get('active_recv', False) and not Work['libE_info'].get('persistent', False):
                        if len(Work['libE_info']['H_rows']) > 0:
                            _, _, _ = self._recv_H_rows(Work)
                        continue

                response = self._handle(Work)
                if response is None:
                    break
                self.comm.send(0, response)

        except Exception as e:
            self.comm.send(0, WorkerErrMsg(' '.join(format_exc_msg(type(e), e)).strip(), format_exc()))
        else:
            self.comm.kill_pending()
        finally:
            self.EnsembleDirectory.copy_back()
