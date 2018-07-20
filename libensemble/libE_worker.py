"""
libEnsemble worker class
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
import os, shutil
import socket

#In future these will be in CalcInfo or Comms modules
#CalcInfo
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.message_numbers import UNSET_TAG, STOP_TAG, CALC_EXCEPTION

#Comms
from libensemble.message_numbers import MAN_SIGNAL_KILL, MAN_SIGNAL_FINISH
from libensemble.message_numbers import MAN_SIGNAL_REQ_RESEND, MAN_SIGNAL_REQ_PICKLE_DUMP

from libensemble.calc_info import CalcInfo
import threading
import logging
from libensemble.controller import JobController
import sys
from libensemble.resources import Resources

wrkid = 'w' + str(Resources.get_workerID())
#logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__ + '(' + wrkid + ')')
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#For debug messages - uncomment
logger.setLevel(logging.DEBUG)

class WorkerException(Exception): pass

#logging to be added
#logging.basicConfig(level=logging.DEBUG,
                    #format='(%(threadName)-10s) %(message)s',
                    #)

#The routine worker_main currently uses MPI. Comms will be implemented using comms module in future
def worker_main(c, sim_specs, gen_specs):
    """ 
    Evaluate calculations given to it by the manager 

    Parameters
    ----------
    c: dict containing fields 'comm' and 'color' for the communicator. 

    sim_specs: dict with parameters/information for simulation calculations

    gen_specs: dict with parameters/information for generation calculations

    """
    
    #Idea is dont have to have it unless using MPI option.
    from mpi4py import MPI
    
    comm = c['comm']
    comm_color = c['color']
    
    rank = comm.Get_rank()
    workerID = rank    
    
    status = MPI.Status()
    Worker.init_workers(sim_specs, gen_specs) # Store in Worker Class
    dtypes = {}
    dtypes[EVAL_SIM_TAG] = None
    dtypes[EVAL_GEN_TAG] = None 
    dtypes[EVAL_SIM_TAG] = comm.bcast(dtypes[EVAL_SIM_TAG] , root=0)
    dtypes[EVAL_GEN_TAG] = comm.bcast(dtypes[EVAL_GEN_TAG] , root=0)

    #worker = Worker(workerID, sim_type, gen_type)
    worker = Worker(workerID)
    
    #Setup logging
    logger.info("Worker {} initiated on MPI rank {} on node {}".format(workerID, rank, socket.gethostname()))
    
    # Print calc_list on-the-fly
    CalcInfo.create_worker_statfile(worker.workerID)
   
    worker_iter = 0
    sim_iter = 0
    gen_iter = 0
    
    #create_exception = this_does_not_exist
    
    #Init in case of manager request before filled
    worker_out={}
    
    while True:
        worker_iter += 1
        logger.debug("Worker {}. Iteration {}".format(workerID,worker_iter))
        
        # General probe for manager communication
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)          
        mtag = status.Get_tag()
        if mtag == STOP_TAG: #If multiple choices prob change this to MANAGER_SIGNAL_TAG or something
            man_signal = comm.recv(source=0, tag=STOP_TAG, status=status)
            if man_signal == MAN_SIGNAL_FINISH: #shutdown the worker
                break
            #Need to handle manager job kill here - as well as finish
            if man_signal == MAN_SIGNAL_REQ_RESEND:               
                #And resend
                logger.debug("Worker {} re-sending to Manager with status {}".format(workerID, worker.calc_status))
                comm.send(obj=worker_out, dest=0)            
                continue
           
            if man_signal == MAN_SIGNAL_REQ_PICKLE_DUMP:
                # Worker is requested to dump pickle file (either for read by manager or for debugging)
                import pickle                
                pfilename="pickled_worker_" + str(workerID) + '_sim_' + str(sim_iter) + '.pkl'
                pickle.dump(worker_out, open(pfilename, "wb"))
                worker_post_pickle_file = pickle.load(open(pfilename, "rb"))  #check can read in this side
                logger.debug("Worker {} dumping pickle and notifying manager: status {}".format(workerID, worker.calc_status))
                comm.send(obj=pfilename, dest=0)
                continue
                
        else:
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
              
        libE_info = Work['libE_info']
        calc_type = Work['tag'] #If send components - send tag separately (dont use MPI.status!)
        
        if calc_type == EVAL_GEN_TAG:
            gen_iter += 1
        elif calc_type == EVAL_SIM_TAG:
            sim_iter += 1
        
        calc_in = np.zeros(len(libE_info['H_rows']),dtype=dtypes[calc_type])
        if len(calc_in) > 0: 
            calc_in = comm.recv(buf=None, source=0)
            logger.debug("Worker {} received calc_in of len {}".format(workerID, np.size(calc_in)))
        
        #This is current kluge for persistent worker - comm will be in the future comms module...
        if 'persistent' in libE_info and libE_info['persistent']:
            libE_info['comm'] = comm
            Work['libE_info'] = libE_info 
                 
        worker.run(Work, calc_in)
        
        if 'persistent' in worker.libE_info and worker.libE_info['persistent']:
            del worker.libE_info['comm']        
        
        CalcInfo.add_calc_worker_statfile(calc = worker.calc_list[-1])    
                
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

        #print("worker {} worker_out: {}".format(workerID,worker_out))
        logger.debug("Worker {} sending to Manager with status {}".format(workerID, worker.calc_status))
        comm.send(obj=worker_out, dest=0) #blocking
        #comm.isend(obj=worker, dest=0) # Non-blocking        
        #comm.send(obj=worker_out, dest=0, tag=worker.calc_type) #blocking
    
    
    if 'clean_jobs' in sim_specs and sim_specs['clean_jobs']:
            worker.clean()
    
    #Destroy worker object?


######################################################################
# Worker Class
######################################################################

# All routines in Worker Class have no MPI and can be called regardless of worker
# concurrency mode.
class Worker():

    #Class attributes    
    sim_specs = {}
    gen_specs = {}
    
    #Class methods
    @staticmethod
    def init_workers(sim_specs_in, gen_specs_in):
        
        #Class attributes? Maybe should be worker specific??
        Worker.sim_specs = sim_specs_in
        Worker.gen_specs = gen_specs_in
        
    #@staticmethod    
    #def get_worker(worker_list,workerID):
        
        #for worker in worker_list:
            #if worker.workerID == workerID:
                #return worker
        ##Does not exist
        #return None
    
    #@staticmethod
    #def get_worker_index(worker_list,workerID):
        
        #index=0
        #for worker in worker_list:
            #if worker.workerID == workerID:
                #return index
            #index += 1
        ##Does not exist
        #return None
    
    #Worker Object methods
    def __init__(self, workerID):

        self.locations = {}
        self.worker_dir = ""
        self.workerID = workerID
        
        self.calc_out = {}
        self.calc_type = None
        self.calc_status = UNSET_TAG #From message_numbers
        self.isdone = False
        self.calc_list = []
        self.job_controller_set = False
        
        #self.sim_specs = Worker.sim_specs
        #self.gen_specs = Worker.gen_specs       

        if 'sim_dir' in Worker.sim_specs:
            #worker_dir = Worker.sim_specs['sim_dir'] + '_' + str(comm_color) + "_" + str(rank) 
            self.worker_dir = Worker.sim_specs['sim_dir'] + '_' + str(self.workerID)
    
            if 'sim_dir_prefix' in Worker.sim_specs:
                self.worker_dir =  os.path.join(os.path.expanduser(Worker.sim_specs['sim_dir_prefix']), os.path.split(os.path.abspath(os.path.expanduser(self.worker_dir)))[1])
    
            assert ~os.path.isdir(self.worker_dir), "Worker directory already exists."
            # if not os.path.exists(worker_dir):
            shutil.copytree(Worker.sim_specs['sim_dir'], self.worker_dir)
            self.locations[EVAL_SIM_TAG] = self.worker_dir
            
        #Optional - set workerID in job_controller - so will be added to jobnames and accesible to calcs
        try:
            jobctl = JobController.controller
            jobctl.set_workerID(workerID)
            #print('workerid',jobctl.workerID)
        except Exception as e:
            logger.info("No job_controller set on worker {}".format(workerID))
            self.job_controller_set = False
        else:
            self.job_controller_set = True


    #worker.run
    def run(self, Work, calc_in):
        
        #Reset run specific attributes - these should maybe be in a calc object
        self.calc_out = {}
        self.calc_type = None
        self.calc_status = UNSET_TAG #From message_numbers
        self.isdone = False  
        self.persis_info = None
        self.libE_info = None
        self.calc_stats = None
        
        # calc_stats stores timing and summary info for this Calc (sim or gen)
        self.calc_stats = CalcInfo()
        self.calc_list.append(self.calc_stats)
        
        #Timing will include setup/teardown
        self.calc_stats.start_timer()
        
        #Could keep all this inside the Work dictionary if sending all Work ...
        self.libE_info = Work['libE_info']
        self.calc_type = Work['tag']
        self.calc_stats.calc_type = Work['tag']
        self.persis_info = Work['persis_info']        
        #logging.debug('Running thread %s on worker %d %s', t.getName(), self.workerID, self.calc_type)
        
        assert self.calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"
        
        #data_out, tag_out = self._perform_calc(calc_in, persis_info, libE_info)
        #data_out, tag_out = self._perform_calc(calc_in)   
        #if self.calc_type == EVAL_SIM_TAG:
            #import pdb; pdb.set_trace()
        
        self.calc_out, self.persis_info, self.libE_info, self.calc_status = self._perform_calc(calc_in, self.persis_info, self.libE_info)
        
        #This is a libe feature that is to be reviewed for best solution
        #Should atleast put in calc_stats.
        self.calc_stats.set_calc_status(self.calc_status)
            
        self.isdone = True
        self.calc_stats.stop_timer()
                
        return # Can retrieve output from worker.data

    
    # Do we want to be removing these dirs by default??? Maybe an option
    # Could be option in sim_specs - "clean_jobdirs"
    def clean(self):
        # Clean up - may need to chdir to saved dir also (saved_dir cld be object attribute)
        for loc in self.locations.values():
            shutil.rmtree(loc)
        return


    #Prob internal only - may make this static - no self vars
    def _perform_calc(self, calc_in, persis_info, libE_info):
        if self.calc_type in self.locations:
            saved_dir = os.getcwd()
            #print('current dir in _perform_calc is ', saved_dir)
            #logging.debug('current dir in _perform_calc is  %s', saved_dir)
            os.chdir(self.locations[self.calc_type])
        
        ### ============================== Run calc ====================================
        # This is in a try/except block to allow handling if exception is raised in user code
        # Currently report exception to summary file and pass exception up (where libE will mpi_abort)
        # Continuation of ensemble may be added as an option.
        if self.calc_type == EVAL_SIM_TAG:
            try:
                out = Worker.sim_specs['sim_f'](calc_in,persis_info,Worker.sim_specs,libE_info)
            except Exception as e:
                # Write to workers summary file and pass exception up
                if self.calc_type in self.locations:
                    os.chdir(saved_dir)                
                self.calc_stats.stop_timer()
                self.calc_status = CALC_EXCEPTION
                self.calc_stats.set_calc_status(self.calc_status)
                CalcInfo.add_calc_worker_statfile(calc = self.calc_stats)
                raise
        else: 
            try:
                out = Worker.gen_specs['gen_f'](calc_in,persis_info,Worker.gen_specs,libE_info)
            except Exception as e:
                # Write to workers summary file and pass exception up
                if self.calc_type in self.locations:
                    os.chdir(saved_dir)
                self.calc_stats.stop_timer()
                self.calc_status = CALC_EXCEPTION
                self.calc_stats.set_calc_status(self.calc_status)
                CalcInfo.add_calc_worker_statfile(calc = self.calc_stats)
                raise            
        ### ============================================================================

        assert isinstance(out, tuple), "Calculation output must be a tuple. Worker exiting"
        assert len(out) >= 2, "Calculation output must be at least two elements when a tuple"

        H = out[0]
        persis_info = out[1]
        
        calc_tag = UNSET_TAG #None
        if len(out) >= 3:
            calc_tag = out[2]

        if self.calc_type in self.locations:
            os.chdir(saved_dir)

        #data_out = {'calc_out':H, 'persis_info':persis_info, 'libE_info': libE_info}
        
        #print('H is:', H)
        
        #return data_out, calc_tag
        return H, persis_info, libE_info, calc_tag
 
