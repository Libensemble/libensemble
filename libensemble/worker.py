"""
libEnsemble worker class
====================================================
"""

import cProfile
import logging
import logging.handlers
import pstats
import socket
from itertools import count
from pathlib import Path
from traceback import format_exc
from traceback import format_exception_only as format_exc_msg

import numpy as np
import numpy.typing as npt

from libensemble.comms.logs import LogConfig, worker_logging_config
from libensemble.executors.executor import Executor
from libensemble.message_numbers import (
    CALC_EXCEPTION,
    EVAL_GEN_TAG,
    EVAL_SIM_TAG,
    MAN_SIGNAL_FINISH,
    MAN_SIGNAL_KILL,
    PERSIS_STOP,
    STOP_TAG,
    UNSET_TAG,
    calc_status_strings,
    calc_type_strings,
)
from libensemble.resources.resources import Resources
from libensemble.utils.loc_stack import LocationStack
from libensemble.utils.misc import extract_H_ranges
from libensemble.utils.output_directory import EnsembleDirectory
from libensemble.utils.runners import Runners
from libensemble.utils.timer import Timer

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)
task_timing = False


def worker_main(
    comm: "Communicator",  # noqa: F821
    sim_specs: dict,
    gen_specs: dict,
    libE_specs: dict,
    workerID: int = None,
    log_comm: bool = True,
    resources: Resources = None,
    executor: Executor = None,
) -> None:  # noqa: F821
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

    if libE_specs.get("profile"):
        pr = cProfile.Profile()
        pr.enable()

    # If resources / executor passed in, then use those.
    if resources is not None:
        Resources.resources = resources
    if executor is not None:
        Executor.executor = executor

    # Receive dtypes from manager
    _, dtypes = comm.recv()

    # Receive workflow dir from manager
    if libE_specs.get("use_workflow_dir"):
        _, libE_specs["workflow_dir_path"] = comm.recv()

    workerID = workerID or comm.rank

    # Initialize logging on comms
    if log_comm:
        worker_logging_config(comm, workerID)

    LS = LocationStack()
    LS.register_loc("workflow", Path(libE_specs.get("workflow_dir_path")))

    # Set up and run worker
    worker = Worker(comm, dtypes, workerID, sim_specs, gen_specs, libE_specs)
    with LS.loc("workflow"):
        worker.run()

    if libE_specs.get("profile"):
        pr.disable()
        profile_state_fname = "worker_%d.prof" % (workerID)

        with open(profile_state_fname, "w") as f:
            ps = pstats.Stats(pr, stream=f).sort_stats("cumulative")
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

    def __init__(
        self,
        comm: "Communicator",  # noqa: F821
        dtypes: npt.DTypeLike,
        workerID: int,
        sim_specs: dict,
        gen_specs: dict,
        libE_specs: dict,
    ) -> None:  # noqa: F821
        """Initializes new worker object"""
        self.comm = comm
        self.dtypes = dtypes
        self.workerID = workerID
        self.libE_specs = libE_specs
        self.stats_fmt = libE_specs.get("stats_fmt", {})

        self.calc_iter = {EVAL_SIM_TAG: 0, EVAL_GEN_TAG: 0}
        self.runners = Runners(sim_specs, gen_specs)
        self._run_calc = self.runners.make_runners()
        Worker._set_executor(self.workerID, self.comm)
        Worker._set_resources(self.workerID, self.comm)
        self.EnsembleDirectory = EnsembleDirectory(libE_specs=libE_specs)

    @staticmethod
    def _set_gen_procs_gpus(libE_info, obj):
        if any(k in libE_info for k in ("num_procs", "num_gpus")):
            obj.set_gen_procs_gpus(libE_info)

    @staticmethod
    def _set_rset_team(libE_info: dict) -> bool:
        """Pass new rset_team to worker resources

        Also passes gen assigned cpus/gpus to resources and executor
        """
        resources = Resources.resources
        exctr = Executor.executor
        if isinstance(resources, Resources):
            wresources = resources.worker_resources
            wresources.set_rset_team(libE_info["rset_team"])
            Worker._set_gen_procs_gpus(libE_info, wresources)
            if isinstance(exctr, Executor):
                Worker._set_gen_procs_gpus(libE_info, exctr)
            return True
        else:
            return False

    @staticmethod
    def _set_executor(workerID: int, comm: "Communicator") -> bool:  # noqa: F821
        """Sets worker ID in the executor, return True if set"""
        exctr = Executor.executor
        if isinstance(exctr, Executor):
            exctr.set_worker_info(comm, workerID)  # When merge update
            return True
        else:
            logger.debug(f"No executor set on worker {workerID}")
            return False

    @staticmethod
    def _set_resources(workerID, comm: "Communicator") -> bool:  # noqa: F821
        """Sets worker ID in the resources, return True if set"""
        resources = Resources.resources
        if isinstance(resources, Resources):
            resources.set_worker_resources(comm.get_num_workers(), workerID)
            return True
        else:
            logger.debug(f"No resources set on worker {workerID}")
            return False

    def _handle_calc(self, Work: dict, calc_in: npt.NDArray) -> (npt.NDArray, dict, int):
        """Runs a calculation on this worker object.

        This routine calls the user calculations. Exceptions are caught,
        dumped to the summary file, and raised.

        Parameters
        ----------

        Work: :obj:`dict`
            :ref:`(example)<datastruct-work-dict>`

        calc_in: obj: numpy structured array
            Rows from the :ref:`history array<funcguides-history>`
            for processing
        """
        calc_type = Work["tag"]
        self.calc_iter[calc_type] += 1

        # calc_stats stores timing and summary info for this Calc (sim or gen)
        # calc_id = next(self._calc_id_counter)

        if calc_type == EVAL_SIM_TAG:
            enum_desc = "sim_id"
            calc_id = extract_H_ranges(Work)
        else:
            enum_desc = "Gen no"
            # Use global gen count if available
            if Work["libE_info"].get("gen_count"):
                calc_id = str(Work["libE_info"]["gen_count"])
            else:
                calc_id = str(self.calc_iter[calc_type])
        # Add a right adjust (minimum width).
        calc_id = calc_id.rjust(5, " ")

        timer = Timer()

        try:
            logger.debug(f"Starting {enum_desc}: {calc_id}")
            calc = self._run_calc[calc_type]
            with timer:
                if self.EnsembleDirectory.use_calc_dirs(calc_type):
                    loc_stack, calc_dir = self.EnsembleDirectory.prep_calc_dir(
                        Work,
                        self.calc_iter,
                        self.workerID,
                        calc_type,
                    )
                    with loc_stack.loc(calc_dir):  # Changes to calculation directory
                        out = calc(calc_in, Work)
                else:
                    out = calc(calc_in, Work)

                logger.debug(f"Returned from user function for {enum_desc} {calc_id}")

            calc_status = UNSET_TAG
            # Check for buffered receive
            if self.comm.recv_buffer:
                tag, message = self.comm.recv()
                if tag in [STOP_TAG, PERSIS_STOP] and message is MAN_SIGNAL_FINISH:
                    calc_status = MAN_SIGNAL_FINISH

            if out:  # better way of doing this logic?
                if len(out) >= 3:  # Out, persis_info, calc_status
                    calc_status = out[2]
                    return out
                elif len(out) == 2:  # Out, persis_info OR Out, calc_status
                    if isinstance(out[1], int) or isinstance(out[1], str):  # got Out, calc_status
                        calc_status = out[1]
                        return out[0], Work["persis_info"], calc_status
                    return *out, calc_status  # got Out, persis_info
                else:
                    return out, Work["persis_info"], calc_status
            else:
                return None, Work["persis_info"], calc_status

        except Exception as e:
            logger.debug(f"Re-raising exception from calc {e}")
            calc_status = CALC_EXCEPTION
            raise
        finally:
            ctype_str = calc_type_strings[calc_type]
            status = calc_status_strings.get(calc_status, calc_status)
            calc_msg = self._get_calc_msg(enum_desc, calc_id, ctype_str, timer, status)

            logging.getLogger(LogConfig.config.stats_name).info(calc_msg)

    def _get_calc_msg(self, enum_desc: str, calc_id: int, calc_type: int, timer: Timer, status: str) -> str:
        """Construct line for libE_stats.txt file"""
        calc_msg = f"{enum_desc} {calc_id}: {calc_type} {timer}"

        if self.stats_fmt.get("task_timing", False) or self.stats_fmt.get("task_datetime", False):
            calc_msg += Executor.executor.new_tasks_timing(datetime=self.stats_fmt.get("task_datetime", False))

        if self.stats_fmt.get("show_resource_sets", False):
            # Maybe just call option resource_sets if already in sub-dictionary
            resources = Resources.resources.worker_resources
            calc_msg += f" rsets: {resources.rset_team}"

        # Always put status last as could involve different numbers of words. Some scripts may assume this.
        calc_msg += f" Status: {status}"

        return calc_msg

    def _recv_H_rows(self, Work: dict) -> (dict, int, npt.NDArray):
        """Unpacks Work request and receives any history rows"""
        libE_info = Work["libE_info"]
        calc_type = Work["tag"]
        if len(libE_info["H_rows"]) > 0:
            _, calc_in = self.comm.recv()
        else:
            calc_in = np.zeros(0, dtype=self.dtypes[calc_type])

        logger.debug(f"Received calc_in ({calc_type_strings[calc_type]}) of len {np.size(calc_in)}")
        assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"

        return libE_info, calc_type, calc_in

    def _handle(self, Work: dict) -> dict:
        """Handles a work request from the manager"""
        # Check work request and receive second message (if needed)
        libE_info, calc_type, calc_in = self._recv_H_rows(Work)

        # Call user function
        libE_info["comm"] = self.comm
        libE_info["workerID"] = self.workerID
        libE_info["rset_team"] = libE_info.get("rset_team", [])
        Worker._set_rset_team(libE_info)

        calc_out, persis_info, calc_status = self._handle_calc(Work, calc_in)

        if "libE_info" in Work:
            libE_info = Work["libE_info"]

        if "comm" in libE_info:
            del libE_info["comm"]

        # If there was a finish signal, bail
        if calc_status == MAN_SIGNAL_FINISH:
            return None

        # Otherwise, send a calc result back to manager
        logger.debug(f"Sending to Manager with status {calc_status}")
        return {
            "calc_out": calc_out,
            "persis_info": persis_info,
            "libE_info": libE_info,
            "calc_status": calc_status,
            "calc_type": calc_type,
        }

    def run(self) -> None:
        """Runs the main worker loop."""
        try:
            logger.info(f"Worker {self.workerID} initiated on node {socket.gethostname()}")

            for worker_iter in count(start=1):
                logger.debug(f"Iteration {worker_iter}")

                mtag, Work = self.comm.recv()

                if mtag in [STOP_TAG, PERSIS_STOP]:
                    if Work is MAN_SIGNAL_FINISH:
                        break
                    elif Work is MAN_SIGNAL_KILL:
                        continue

                # Active recv is for persistent worker only - throw away here
                if isinstance(Work, dict):
                    if Work.get("libE_info", False):
                        if Work["libE_info"].get("active_recv", False) and not Work["libE_info"].get(
                            "persistent", False
                        ):
                            if len(Work["libE_info"]["H_rows"]) > 0:
                                _, _, _ = self._recv_H_rows(Work)
                            continue
                else:
                    logger.debug(f"mtag: {mtag}; Work: {Work}")
                    raise

                response = self._handle(Work)
                if response is None:
                    break
                self.comm.send(0, response)

        except Exception as e:
            self.comm.send(0, WorkerErrMsg(" ".join(format_exc_msg(type(e), e)).strip(), format_exc()))
        else:
            self.comm.kill_pending()
        finally:
            self.runners.shutdown()
            self.EnsembleDirectory.copy_back()
