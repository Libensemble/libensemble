"""
libEnsemble worker class
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import socket
import logging
import numpy as np

from mpi4py import MPI

from libensemble.message_numbers import \
     EVAL_SIM_TAG, EVAL_GEN_TAG, \
     UNSET_TAG, STOP_TAG, CALC_EXCEPTION
from libensemble.message_numbers import \
     MAN_SIGNAL_FINISH, \
     MAN_SIGNAL_REQ_RESEND, MAN_SIGNAL_REQ_PICKLE_DUMP

from libensemble.loc_stack import LocationStack
from libensemble.calc_info import CalcInfo
from libensemble.controller import JobController
from libensemble.resources import Resources

logger = logging.getLogger(__name__ + '(' + Resources.get_my_name() + ')')
#For debug messages in this module  - uncomment (see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)


def recv_dtypes(comm):
    """Receive dtypes array broadcast from manager."""
    dtypes = {}
    dtypes[EVAL_SIM_TAG] = None
    dtypes[EVAL_GEN_TAG] = None
    dtypes[EVAL_SIM_TAG] = comm.bcast(dtypes[EVAL_SIM_TAG], root=0)
    dtypes[EVAL_GEN_TAG] = comm.bcast(dtypes[EVAL_GEN_TAG], root=0)
    return dtypes


#The routine worker_main currently uses MPI. Comms will be implemented using comms module in future
def worker_main(c, sim_specs, gen_specs):
    """
    Evaluate calculations given to it by the manager.

    Creates a worker object, receives work from manager, runs worker,
    and communicates results. This routine also creates and writes to
    the workers summary file.

    Parameters
    ----------
    c: dict containing fields 'comm' and 'color' for the communicator.

    sim_specs: dict with parameters/information for simulation calculations

    gen_specs: dict with parameters/information for generation calculations

    """

    comm = c['comm']

    rank = comm.Get_rank()
    workerID = rank

    status = MPI.Status()
    dtypes = recv_dtypes(comm)

    worker = Worker(workerID, sim_specs, gen_specs)

    #Setup logging
    logger.info("Worker {} initiated on MPI rank {} on node {}".format(workerID, rank, socket.gethostname()))

    # Print calc_list on-the-fly
    CalcInfo.create_worker_statfile(worker.workerID)

    worker_iter = 0
    calc_iter = {EVAL_GEN_TAG : 0, EVAL_SIM_TAG : 0}

    #Init in case of manager request before filled
    worker_out = {}

    while True:
        worker_iter += 1
        logger.debug("Worker {}. Iteration {}".format(workerID, worker_iter))

        # Receive message from worker
        msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        mtag = status.Get_tag()
        if mtag == STOP_TAG: #If multiple choices prob change this to MANAGER_SIGNAL_TAG or something
            if msg == MAN_SIGNAL_FINISH: #shutdown the worker
                break
            #Need to handle manager job kill here - as well as finish
            if msg == MAN_SIGNAL_REQ_RESEND:
                logger.debug("Worker {} re-sending to Manager with status {}".format(workerID, worker.calc_status))
                comm.send(obj=worker_out, dest=0)
                continue

            if msg == MAN_SIGNAL_REQ_PICKLE_DUMP:
                # Worker is requested to dump pickle file (either for read by manager or for debugging)
                import pickle
                pfilename = "pickled_worker_{}_sim_{}.pkl".format(workerID, calc_iter[EVAL_SIM_TAG])
                with open(pfilename, "wb") as f:
                    pickle.dump(worker_out, f)
                with open(pfilename, "rb") as f:
                    pickle.load(f)  #check can read in this side
                logger.debug("Worker {} dumping pickle and notifying manager: status {}".format(workerID, worker.calc_status))
                comm.send(obj=pfilename, dest=0)
                continue

        Work = msg
        libE_info = Work['libE_info']
        calc_type = Work['tag'] #If send components - send tag separately (dont use MPI.status!)
        calc_iter[calc_type] += 1

        calc_in = (comm.recv(source=0) if len(libE_info['H_rows']) > 0
                   else np.zeros(0, dtype=dtypes[calc_type]))
        logger.debug("Worker {} received calc_in of len {}".format(workerID, np.size(calc_in)))

        #This is current kluge for persistent worker - comm will be in the future comms module...
        if libE_info.get('persistent'):
            libE_info['comm'] = comm
            Work['libE_info'] = libE_info

        worker.run(Work, calc_in)

        if worker.libE_info.get('persistent'):
            del worker.libE_info['comm']

        #Check if sim/gen func recieved a finish signal...
        #Currently this means do not send data back first
        if worker.calc_status == MAN_SIGNAL_FINISH:
            break

        # Determine data to be returned to manager
        worker_out = {'calc_out': worker.calc_out,
                      'persis_info': worker.persis_info,
                      'libE_info': worker.libE_info,
                      'calc_status': worker.calc_status,
                      'calc_type': worker.calc_type}

        logger.debug("Worker {} sending to Manager with status {}".format(workerID, worker.calc_status))
        comm.send(obj=worker_out, dest=0) #blocking

    if sim_specs.get('clean_jobs'):
        worker.clean()


######################################################################
# Worker Class
######################################################################

class Worker():

    """The Worker Class provides methods for controlling sim and gen funcs"""

    # Worker Object methods
    def __init__(self, workerID, sim_specs, gen_specs):
        """Initialise new worker object.

        Parameters
        ----------

        workerID: int:
            The ID for this worker

        """

        self.workerID = workerID

        self.calc_out = {}
        self.calc_type = None
        self.calc_status = UNSET_TAG #From message_numbers
        self.calc_list = []

        self.persis_info = None
        self.libE_info = None

        self._run_calc = Worker._make_runners(sim_specs, gen_specs)
        self.loc_stack = self._make_sim_worker_dir(sim_specs)
        self._set_job_controller()


    def _make_sim_worker_dir(self, sim_specs, locs=None):
        "Create a dir for sim workers if 'sim_dir' is in sim_specs"
        locs = locs or LocationStack()
        if 'sim_dir' in sim_specs:
            sim_dir = sim_specs['sim_dir']
            prefix = sim_specs.get('sim_dir_prefix')
            worker_dir = "{}_{}".format(sim_dir, self.workerID)
            locs.register_loc(EVAL_SIM_TAG, worker_dir,
                              prefix=prefix, srcdir=sim_dir)
        return locs


    @staticmethod
    def _make_runners(sim_specs, gen_specs):
        "Create functions to run a sim or gen"

        def run_sim(calc_in, persis_info, libE_info):
            "Run a sim calculation"
            return sim_specs['sim_f'](calc_in, persis_info, sim_specs, libE_info)

        def run_gen(calc_in, persis_info, libE_info):
            "Run a gen calculation"
            return gen_specs['gen_f'](calc_in, persis_info, gen_specs, libE_info)

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}


    def _set_job_controller(self):
        "Optional -- set worker ID in the job controller, return if set"
        try:
            jobctl = JobController.controller
            jobctl.set_workerID(self.workerID)
        except Exception:
            logger.info("No job_controller set on worker {}".format(self.workerID))
            return False
        else:
            return True


    def run(self, Work, calc_in):
        """Run a calculation on this worker object.

        This routine calls the user calculations. Exceptions are caught, dumped to
        the summary file, and raised.

        Parameters
        ----------

        Work: :obj:`dict`
            :ref:`(example)<datastruct-work-dict>`

        calc_in: obj: numpy structured array
            Rows from the :ref:`history array<datastruct-history-array>` for processing

        """
        assert Work['tag'] in [EVAL_SIM_TAG, EVAL_GEN_TAG], \
          "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"

        # calc_stats stores timing and summary info for this Calc (sim or gen)
        calc_stats = CalcInfo()
        self.calc_list.append(calc_stats)

        #Timing will include setup/teardown
        calc_stats.start_timer()

        #Could keep all this inside the Work dictionary if sending all Work ...
        self.libE_info = Work['libE_info']
        self.calc_type = Work['tag']
        self.persis_info = Work['persis_info']
        calc_stats.calc_type = self.calc_type

        try:
            with self.loc_stack.loc(self.calc_type):
                out = self._run_calc[self.calc_type](calc_in, self.persis_info, self.libE_info)
            assert isinstance(out, tuple), "Calculation output must be a tuple. Worker exiting"
            assert len(out) >= 2, "Calculation output must be at least two elements when a tuple"
            self.calc_out = out[0]
            self.persis_info = out[1]
            self.calc_status = out[2] if len(out) >= 3 else UNSET_TAG
        except Exception as e:
            self.calc_out = {}
            self.calc_status = CALC_EXCEPTION
            raise
        finally:
            calc_stats.stop_timer()
            calc_stats.set_calc_status(self.calc_status)
            CalcInfo.add_calc_worker_statfile(calc=calc_stats)


    def clean(self):
        """Clean up calculation directories"""
        self.loc_stack.clean_locs()
