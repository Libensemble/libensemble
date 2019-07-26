"""
libEnsemble worker class
====================================================
"""

import socket
import logging
import logging.handlers
from itertools import count
from traceback import format_exc

import numpy as np

from libensemble.message_numbers import \
    EVAL_SIM_TAG, EVAL_GEN_TAG, \
    UNSET_TAG, STOP_TAG, CALC_EXCEPTION
from libensemble.message_numbers import MAN_SIGNAL_FINISH
from libensemble.message_numbers import calc_type_strings, calc_status_strings

from libensemble.util.loc_stack import LocationStack
from libensemble.util.timer import Timer
from libensemble.controller import JobController
from libensemble.comms.logs import worker_logging_config
from libensemble.comms.logs import LogConfig

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)
job_timing = False


def worker_main(comm, sim_specs, gen_specs, workerID=None, log_comm=True):
    """Evaluate calculations given to it by the manager.

    Creates a worker object, receives work from manager, runs worker,
    and communicates results. This routine also creates and writes to
    the workers summary file.

    Parameters
    ----------
    comm: comm
        Comm object for manager communications

    sim_specs: dict
        Parameters/information for simulation calculations

    gen_specs: dict
        Parameters/information for generation calculations

    workerID: int
        Manager assigned worker ID (if None, default is comm.rank)

    log_comm: boolean
        Whether to send logging over comm
    """

    # Receive dtypes from manager
    _, dtypes = comm.recv()
    workerID = workerID or comm.rank

    # Initialize logging on comms
    if log_comm:
        worker_logging_config(comm, workerID)

    # Set up and run worker
    worker = Worker(comm, dtypes, workerID, sim_specs, gen_specs)
    worker.run()


######################################################################
# Worker Class
######################################################################


class WorkerErrMsg:

    def __init__(self, msg, exc):
        self.msg = msg
        self.exc = exc


class Worker:

    """The Worker Class provides methods for controlling sim and gen funcs

    **Object Attributes:**

    These are public object attributes.

    :ivar comm comm:
        Comm object for manager communications

    :ivar dict dtypes:
        Dictionary containing type information for sim and gen inputs

    :ivar int workerID:
        The libensemble Worker ID

    :ivar dict sim_specs:
        Parameters/information for simulation calculations

    :ivar dict calc_iter:
        Dictionary containing counts for each type of calc (e.g. sim or gen)

    :ivar LocationStack loc_stack:
        Stack holding directory structure of this Worker
    """

    def __init__(self, comm, dtypes, workerID, sim_specs, gen_specs):
        """Initialise new worker object.

        """
        self.comm = comm
        self.dtypes = dtypes
        self.workerID = workerID
        self.sim_specs = sim_specs
        self.calc_iter = {EVAL_SIM_TAG: 0, EVAL_GEN_TAG: 0}
        self.loc_stack = None  # Worker._make_sim_worker_dir(sim_specs, workerID)
        self._run_calc = Worker._make_runners(sim_specs, gen_specs)
        self._calc_id_counter = count()
        Worker._set_job_controller(self.workerID, self.comm)

    @staticmethod
    def _make_sim_worker_dir(sim_specs, workerID, locs=None):
        "Create a dir for sim workers if 'sim_dir' is in sim_specs"
        locs = locs or LocationStack()
        if 'sim_dir' in sim_specs:
            sim_dir = sim_specs['sim_dir'].rstrip('/')
            prefix = sim_specs.get('sim_dir_prefix')
            worker_dir = "{}_worker{}".format(sim_dir, workerID)
            locs.register_loc(EVAL_SIM_TAG, worker_dir,
                              prefix=prefix, srcdir=sim_dir)
        return locs

    @staticmethod
    def _make_runners(sim_specs, gen_specs):
        "Create functions to run a sim or gen"

        sim_f = sim_specs['sim_f']

        def run_sim(calc_in, persis_info, libE_info):
            "Call the sim func."
            return sim_f(calc_in, persis_info, sim_specs, libE_info)

        if gen_specs:
            gen_f = gen_specs['gen_f']

            def run_gen(calc_in, persis_info, libE_info):
                "Call the gen func."
                return gen_f(calc_in, persis_info, gen_specs, libE_info)
        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    @staticmethod
    def _set_job_controller(workerID, comm):
        "Optional -- set worker ID in the job controller, return if set"
        jobctl = JobController.controller
        if isinstance(jobctl, JobController):
            jobctl.set_worker_info(comm, workerID)
            return True
        else:
            logger.info("No job_controller set on worker {}".format(workerID))
            return False

    def _handle_calc(self, Work, calc_in):
        """Run a calculation on this worker object.

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
        calc_id = next(self._calc_id_counter)
        timer = Timer()

        try:
            logger.debug("Running {}".format(calc_type_strings[calc_type]))
            calc = self._run_calc[calc_type]
            with timer:
                logger.debug("Calling calc {}".format(calc_type))

                # Worker creates own sim_dir only if sim work performed.
                if calc_type == EVAL_SIM_TAG and self.loc_stack:
                    with self.loc_stack.loc(calc_type):
                        out = calc(calc_in, Work['persis_info'], Work['libE_info'])

                elif calc_type == EVAL_SIM_TAG and not self.loc_stack:
                    self.loc_stack = Worker._make_sim_worker_dir(self.sim_specs, self.workerID)
                    with self.loc_stack.loc(calc_type):
                        out = calc(calc_in, Work['persis_info'], Work['libE_info'])

                else:
                    out = calc(calc_in, Work['persis_info'], Work['libE_info'])

                logger.debug("Return from calc call")

            assert isinstance(out, tuple), \
                "Calculation output must be a tuple."
            assert len(out) >= 2, \
                "Calculation output must be at least two elements."

            calc_status = out[2] if len(out) >= 3 else UNSET_TAG
            return out[0], out[1], calc_status
        except Exception:
            logger.debug("Re-raising exception from calc")
            calc_status = CALC_EXCEPTION
            raise
        finally:
            # This was meant to be handled by calc_stats module.
            if job_timing and JobController.controller.list_of_jobs:
                # Initially supporting one per calc. One line output.
                job = JobController.controller.list_of_jobs[-1]
                calc_msg = "Calc {:5d}: {} {} {} Status: {}".\
                    format(calc_id,
                           calc_type_strings[calc_type],
                           timer,
                           job.timer,
                           calc_status_strings.get(calc_status, "Completed"))
            else:
                calc_msg = "Calc {:5d}: {} {} Status: {}".\
                    format(calc_id,
                           calc_type_strings[calc_type],
                           timer,
                           calc_status_strings.get(calc_status, "Completed"))

            logging.getLogger(LogConfig.config.stats_name).info(calc_msg)

    def _recv_H_rows(self, Work):
        "Unpack Work request and receiv any history rows we need."

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
        "Handle a work request from the manager."

        # Check work request and receive second message (if needed)
        libE_info, calc_type, calc_in = self._recv_H_rows(Work)

        # Call user function
        libE_info['comm'] = self.comm
        calc_out, persis_info, calc_status = self._handle_calc(Work, calc_in)
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
        "Run the main worker loop."

        try:
            logger.info("Worker {} initiated on node {}".
                        format(self.workerID, socket.gethostname()))

            for worker_iter in count(start=1):
                logger.debug("Iteration {}".format(worker_iter))

                mtag, Work = self.comm.recv()
                if mtag == STOP_TAG:
                    break

                response = self._handle(Work)
                if response is None:
                    break
                self.comm.send(0, response)

        except Exception as e:
            self.comm.send(0, WorkerErrMsg(str(e), format_exc()))
        else:
            self.comm.kill_pending()
        finally:
            if self.sim_specs.get('clean_jobs') and self.loc_stack is not None:
                self.loc_stack.clean_locs()
