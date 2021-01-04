"""
libEnsemble worker class
====================================================
"""

import socket
import logging
import os
import shutil
import re
import logging.handlers
from itertools import count, groupby
from operator import itemgetter
from traceback import format_exc

import numpy as np

from libensemble.message_numbers import \
    EVAL_SIM_TAG, EVAL_GEN_TAG, \
    UNSET_TAG, STOP_TAG, PERSIS_STOP, CALC_EXCEPTION
from libensemble.message_numbers import MAN_SIGNAL_FINISH
from libensemble.message_numbers import calc_type_strings, calc_status_strings
from libensemble.tools.fields_keys import libE_spec_sim_dir_keys, libE_spec_gen_dir_keys, libE_spec_calc_dir_misc

from libensemble.utils.loc_stack import LocationStack
from libensemble.utils.timer import Timer
from libensemble.executors.executor import Executor
from libensemble.comms.logs import worker_logging_config
from libensemble.comms.logs import LogConfig
import cProfile
import pstats

if tuple(np.__version__.split('.')) >= ('1', '15'):
    from numpy.lib.recfunctions import repack_fields

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

    :ivar LocationStack loc_stack:
        Stack holding directory structure of this Worker
    """

    def __init__(self, comm, dtypes, workerID, sim_specs, gen_specs, libE_specs):
        """Initializes new worker object

        """
        self.comm = comm
        self.dtypes = dtypes
        self.workerID = workerID
        self.sim_specs = sim_specs
        self.libE_specs = libE_specs
        self.startdir = os.getcwd()
        self.prefix = libE_specs.get('ensemble_dir_path', './ensemble')
        self.calc_iter = {EVAL_SIM_TAG: 0, EVAL_GEN_TAG: 0}
        self.loc_stack = None
        self._run_calc = Worker._make_runners(sim_specs, gen_specs)
        self._calc_id_counter = count()
        Worker._set_executor(self.workerID, self.comm)

    @staticmethod
    def _make_calc_dir(libE_specs, workerID, H_rows, calc_str, locs):
        "Create calc dirs and intermediate dirs, copy inputs, based on libE_specs"

        if calc_str == 'sim':
            calc_input_dir = libE_specs.get('sim_input_dir', '').rstrip('/')
            do_calc_dirs = libE_specs.get('sim_dirs_make', False)
            copy_files = libE_specs.get('sim_dir_copy_files', [])
            symlink_files = libE_specs.get('sim_dir_symlink_files', [])
        else:  # calc_str is 'gen'
            calc_input_dir = libE_specs.get('gen_input_dir', '').rstrip('/')
            do_calc_dirs = libE_specs.get('gen_dirs_make', False)
            copy_files = libE_specs.get('gen_dir_copy_files', [])
            symlink_files = libE_specs.get('gen_dir_symlink_files', [])

        prefix = libE_specs.get('ensemble_dir_path', './ensemble')
        do_work_dirs = libE_specs.get('use_worker_dirs', False)

        # If 'use_worker_dirs' only calc_dir option. Use worker dirs, but no calc dirs
        if do_work_dirs and not libE_specs.get('sim_dirs_make') and not libE_specs.get('gen_dirs_make'):
            do_calc_dirs = False

        # If using calc_input_dir, set of files to copy is contents of provided dir
        if calc_input_dir:
            copy_files = set(copy_files + [os.path.join(calc_input_dir, i) for i in os.listdir(calc_input_dir)])

        # If identical paths to copy and symlink, remove those paths from symlink_files
        if len(symlink_files):
            symlink_files = [i for i in symlink_files if i not in copy_files]

        # Cases where individual sim_dirs not created.
        if not do_calc_dirs:
            if do_work_dirs:  # Each worker does work in worker dirs
                key = workerID
                dir = "worker" + str(workerID)
            else:  # Each worker does work in prefix (ensemble_dir)
                key = prefix
                dir = prefix
                prefix = None

            locs.register_loc(key, dir, prefix=prefix, copy_files=copy_files,
                              symlink_files=symlink_files, ignore_FileExists=True)
            return key

        # All cases now should involve sim_dirs
        # ensemble_dir/worker_dir registered here, set as parent dir for sim dirs
        if do_work_dirs:
            worker_dir = "worker" + str(workerID)
            worker_path = os.path.abspath(os.path.join(prefix, worker_dir))
            calc_dir = calc_str + str(H_rows)
            locs.register_loc(workerID, worker_dir, prefix=prefix)
            calc_prefix = worker_path

        # Otherwise, ensemble_dir set as parent dir for sim dirs
        else:
            calc_dir = "{}{}_worker{}".format(calc_str, H_rows, workerID)
            if not os.path.isdir(prefix):
                os.makedirs(prefix, exist_ok=True)
            calc_prefix = prefix

        # Register calc dir with adjusted parent dir and source-file location
        locs.register_loc(calc_dir, calc_dir,  # Dir name also label in loc stack dict
                          prefix=calc_prefix,
                          copy_files=copy_files,
                          symlink_files=symlink_files)

        return calc_dir

    @staticmethod
    def _make_runners(sim_specs, gen_specs):
        "Creates functions to run a sim or gen"

        sim_f = sim_specs['sim_f']

        def run_sim(calc_in, persis_info, libE_info):
            "Calls the sim func."
            return sim_f(calc_in, persis_info, sim_specs, libE_info)

        if gen_specs:
            gen_f = gen_specs['gen_f']

            def run_gen(calc_in, persis_info, libE_info):
                "Calls the gen func."
                return gen_f(calc_in, persis_info, gen_specs, libE_info)
        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    @staticmethod
    def _set_executor(workerID, comm):
        "Optional - sets worker ID in the executor, return if set"
        exctr = Executor.executor
        if isinstance(exctr, Executor):
            exctr.set_worker_info(comm, workerID)
            return True
        else:
            logger.info("No executor set on worker {}".format(workerID))
            return False

    @staticmethod
    def _extract_H_ranges(Work):
        """ Convert received H_rows into ranges for logging, labeling """
        work_H_rows = Work['libE_info']['H_rows']
        if len(work_H_rows) == 1:
            return str(work_H_rows[0])
        else:
            # From https://stackoverflow.com/a/30336492
            # Create groups by difference between row values and sequential enumerations:
            # e.g., [2, 3, 5, 6] -> [(0, 2), (1, 3), (2, 5), (3, 6)]
            #  -> diff=-2, group=[(0, 2), (1, 3)], diff=-3, group=[(2, 5), (3, 6)]
            ranges = []
            for diff, group in groupby(enumerate(work_H_rows.tolist()), lambda x: x[0]-x[1]):
                # Take second values (rows values) from each group element into lists:
                # group=[(0, 2), (1, 3)], group=[(2, 5), (3, 6)] -> group=[2, 3], group=[5, 6]
                group = list(map(itemgetter(1), group))
                if len(group) > 1:
                    ranges.append(str(group[0]) + '-' + str(group[-1]))
                else:
                    ranges.append(str(group[0]))
            return '_'.join(ranges)

    def _copy_back(self):
        """Copy back all ensemble dir contents to launch location"""
        if os.path.isdir(self.prefix) and self.libE_specs.get('ensemble_copy_back', False):

            no_calc_dirs = not self.libE_specs.get('sim_dirs_make', True) or \
                not self.libE_specs.get('gen_dirs_make', True)

            ensemble_dir_path = self.libE_specs.get('ensemble_dir_path', './ensemble')
            copybackdir = os.path.basename(ensemble_dir_path)

            if os.path.relpath(ensemble_dir_path) == os.path.relpath(copybackdir):
                copybackdir += '_back'

            for dir in self.loc_stack.dirs.values():
                dest_path = os.path.join(copybackdir, os.path.basename(dir))
                if dir == self.prefix:  # occurs when no_calc_dirs is True
                    continue  # otherwise, entire ensemble dir copied into copyback dir

                shutil.copytree(dir, dest_path, symlinks=True)
                if os.path.basename(dir).startswith('worker'):
                    return  # Worker dir (with all contents) has been copied.

            # If not using calc dirs, likely miscellaneous files to copy back
            if no_calc_dirs:
                p = re.compile(r"((^sim)|(^gen))\d+_worker\d+")
                for file in [i for i in os.listdir(self.prefix) if not p.match(i)]:  # each non-calc_dir file
                    source_path = os.path.join(self.prefix, file)
                    dest_path = os.path.join(copybackdir, file)
                    try:
                        if os.path.isdir(source_path):
                            shutil.copytree(source_path, dest_path, symlinks=True)
                        else:
                            shutil.copy(source_path, dest_path, follow_symlinks=False)
                    except FileExistsError:
                        continue
                    except shutil.SameFileError:  # creating an identical symlink
                        continue

    def _determine_dir_then_calc(self, Work, calc_type, calc_in, calc):
        "Determines choice for calc_dir structure, then performs calculation."

        if not self.loc_stack:
            self.loc_stack = LocationStack()

        if calc_type == EVAL_SIM_TAG:
            H_rows = Worker._extract_H_ranges(Work)
        else:
            H_rows = str(self.calc_iter[calc_type])

        calc_str = calc_type_strings[calc_type]

        calc_dir = Worker._make_calc_dir(self.libE_specs, self.workerID,
                                         H_rows, calc_str, self.loc_stack)

        with self.loc_stack.loc(calc_dir):  # Switching to calc_dir
            return calc(calc_in, Work['persis_info'], Work['libE_info'])

    def _use_calc_dirs(self, type):
        "Determines calc_dirs enabling for each calc type"

        if type == EVAL_SIM_TAG:
            dir_type_keys = libE_spec_sim_dir_keys
        else:
            dir_type_keys = libE_spec_gen_dir_keys

        dir_type_keys += libE_spec_calc_dir_misc

        return any([setting in self.libE_specs for setting in dir_type_keys])

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
        calc_id = next(self._calc_id_counter)
        timer = Timer()

        try:
            logger.debug("Running {}".format(calc_type_strings[calc_type]))
            calc = self._run_calc[calc_type]
            with timer:
                logger.debug("Calling calc {}".format(calc_type))

                if self._use_calc_dirs(calc_type):
                    out = self._determine_dir_then_calc(Work, calc_type, calc_in, calc)
                else:
                    out = calc(calc_in, Work['persis_info'], Work['libE_info'])

                logger.debug("Return from calc call")

            assert isinstance(out, tuple), \
                "Calculation output must be a tuple."
            assert len(out) >= 2, \
                "Calculation output must be at least two elements."

            calc_status = out[2] if len(out) >= 3 else UNSET_TAG

            # Check for buffered receive
            if self.comm.recv_buffer:
                tag, message = self.comm.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    if message is MAN_SIGNAL_FINISH:
                        calc_status = MAN_SIGNAL_FINISH

            return out[0], out[1], calc_status
        except Exception:
            logger.debug("Re-raising exception from calc")
            calc_status = CALC_EXCEPTION
            raise
        finally:
            # This was meant to be handled by calc_stats module.
            if task_timing and Executor.executor.list_of_tasks:
                # Initially supporting one per calc. One line output.
                task = Executor.executor.list_of_tasks[-1]
                calc_msg = "Calc {:5d}: {} {} {} Status: {}".\
                    format(calc_id,
                           calc_type_strings[calc_type],
                           timer,
                           task.timer,
                           calc_status_strings.get(calc_status, "Not set"))
            else:
                calc_msg = "Calc {:5d}: {} {} Status: {}".\
                    format(calc_id,
                           calc_type_strings[calc_type],
                           timer,
                           calc_status_strings.get(calc_status, "Not set"))

            logging.getLogger(LogConfig.config.stats_name).info(calc_msg)

    def _recv_H_rows(self, Work):
        "Unpacks Work request and receives any history rows"

        libE_info = Work['libE_info']
        calc_type = Work['tag']
        if len(libE_info['H_rows']) > 0:
            _, calc_in = self.comm.recv()
        else:
            if 'repack_fields' in globals():
                calc_in = repack_fields(np.zeros(0, dtype=self.dtypes[calc_type]), recurse=True)
            else:
                calc_in = np.zeros(0, dtype=self.dtypes[calc_type])

        logger.debug("Received calc_in ({}) of len {}".
                     format(calc_type_strings[calc_type], np.size(calc_in)))
        assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], \
            "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"

        return libE_info, calc_type, calc_in

    def _handle(self, Work):
        "Handles a work request from the manager"

        # Check work request and receive second message (if needed)
        libE_info, calc_type, calc_in = self._recv_H_rows(Work)

        # Call user function
        libE_info['comm'] = self.comm
        libE_info['workerID'] = self.workerID
        # libE_info['worker_team'] = [self.workerID] + libE_info.get('blocking', [])
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
        "Runs the main worker loop."

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
            self._copy_back()  # Copy back current results on Exception
        else:
            self.comm.kill_pending()
        finally:
            self._copy_back()
