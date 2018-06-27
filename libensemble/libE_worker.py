"""
libEnsemble worker class
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
import os, shutil
import socket
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.message_numbers import UNSET_TAG, STOP_TAG
from libensemble.message_numbers import MAN_SIGNAL_KILL, MAN_SIGNAL_FINISH
from libensemble.calc_info import CalcInfo
import threading
import logging
from libensemble.controller import JobController

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
    sim_type = comm.recv(buf=None, source=0)
    gen_type = comm.recv(buf=None, source=0)
    dtypes = {}
    dtypes[EVAL_SIM_TAG] = sim_type
    dtypes[EVAL_GEN_TAG] = gen_type

    #worker = Worker(workerID, sim_type, gen_type)
    worker = Worker(workerID)
    
    #Setup logging
    print('Worker %d initiated on MPI rank %d on node %s' % (workerID, rank, socket.gethostname()))
    
    # Print calc_list on-the-fly
    CalcInfo.create_worker_statfile(worker.workerID)
    #timing_file = CalcInfo.stat_file + '.w' + str(worker.workerID)
    #with open(timing_file,'w', buffering=1) as f:
        #f.write("Worker %d:\n" % (worker.workerID))

    while True:
        
        # General probe for manager communication
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)          
        mtag = status.Get_tag()
        if mtag == STOP_TAG:
            man_signal = comm.recv(source=0, tag=STOP_TAG, status=status)
            if man_signal == MAN_SIGNAL_FINISH: #shutdown the worker
                break
            #Need to handle manager job kill here - as well as finish
        else:
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
              
        libE_info = Work['libE_info']
        calc_type = Work['tag'] #If send components - send tag separately (dont use MPI.status!)
        calc_in = np.zeros(len(libE_info['H_rows']),dtype=dtypes[calc_type])
        if len(calc_in) > 0: 
            calc_in = comm.recv(buf=None, source=0)   
        
        #This is current kluge for persistent worker - comm will be in the future comms module...
        if 'persistent' in libE_info and libE_info['persistent']:
            libE_info['comm'] = comm
            Work['libE_info'] = libE_info 
                 
        worker.run(Work, calc_in)
        
        if 'persistent' in worker.libE_info and worker.libE_info['persistent']:
            del worker.libE_info['comm']        
        
        CalcInfo.add_calc_worker_statfile(workerID = worker.workerID, calc = worker.calc_list[-1])
        #with open(timing_file,'a') as f:
            #worker.calc_list[-1].print_calc(f)     
                
        #Check if sim/gen func recieved a finish signal...
        #Currently this means do not send data back first
        if worker.calc_status == MAN_SIGNAL_FINISH:
            break
        
        # Determine data to be returned to manager
        worker_out = {'calc_out': worker.calc_out,
                      'gen_info': worker.gen_info,
                      'libE_info': worker.libE_info,
                      'calc_status': worker.calc_status,
                      'calc_type': worker.calc_type}
        
        comm.send(obj=worker_out, dest=0) #blocking
        #comm.isend(obj=worker, dest=0) # Non-blocking        
        #comm.send(obj=worker_out, dest=0, tag=worker.calc_type) #blocking
    
    
    if 'clean_jobs' in sim_specs and sim_specs['clean_jobs']:
            worker.clean()
    
    ## Print calc_list here
    #timing_file = 'timing.dat.w' + str(worker.workerID)
    #with open(timing_file,'w') as f:
        #f.write("Worker %d:\n" % (worker.workerID))
        #for jb in worker.calc_list:            
            #jb.print_calc(f)
    
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
            
        #Optional - set workerID in job_controller - so will be added to jobnames
        try:
            jobctl = JobController.controller
            jobctl.set_workerID(workerID)
            print('workerid',jobctl.workerID)
        except Exception as e:
            #logger
            print("Info: No job_controller set on worker", workerID)
            self.job_controller_set = False
        else:
            self.job_controller_set = True
            #jobctl.set_workerID(workerID)


    #worker.run
    def run(self, Work, calc_in):
        
        #Reset run specific attributes - these should maybe be in a calc object
        self.calc_out = {}
        self.calc_type = None
        self.calc_status = UNSET_TAG #From message_numbers
        self.isdone = False  
        self.gen_info = None
        self.libE_info = None
        
        # calc_stats stores timing and summary info for this Calc (sim or gen)
        calc_stats = CalcInfo()
        self.calc_list.append(calc_stats)
        
        #Timing will include setup/teardown
        calc_stats.start_timer()
        
        #Could keep all this inside the Work dictionary if sending all Work ...
        self.libE_info = Work['libE_info']
        self.calc_type = Work['tag']
        calc_stats.calc_type = Work['tag']
        self.gen_info = Work['gen_info']        
        #logging.debug('Running thread %s on worker %d %s', t.getName(), self.workerID, self.calc_type)
        
        assert self.calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"
        
        #data_out, tag_out = self._perform_calc(calc_in, gen_info, libE_info)
        #data_out, tag_out = self._perform_calc(calc_in)   
        
        self.calc_out, self.gen_info, self.libE_info, self.calc_status = self._perform_calc(calc_in, self.gen_info, self.libE_info)
        
        #This is a libe feature that is to be reviewed for best solution
        #Should atleast put in calc_stats.
        calc_stats.set_calc_status(self.calc_status)
            
        self.isdone = True
        calc_stats.stop_timer()
                
        return # Can retrieve output from worker.data

    
    # Do we want to be removing these dirs by default??? Maybe an option
    # Could be option in sim_specs - "clean_jobdirs"
    def clean(self):
        # Clean up - may need to chdir to saved dir also (saved_dir cld be object attribute)
        for loc in self.locations.values():
            shutil.rmtree(loc)
        return


    #Prob internal only - may make this static - no self vars
    def _perform_calc(self, calc_in, gen_info, libE_info):
        if self.calc_type in self.locations:
            saved_dir = os.getcwd()
            #print('current dir in _perform_calc is ', saved_dir)
            #logging.debug('current dir in _perform_calc is  %s', saved_dir)
            os.chdir(self.locations[self.calc_type])
        
        ### ============================== Run calc ====================================
        if self.calc_type == EVAL_SIM_TAG:
            out = Worker.sim_specs['sim_f'](calc_in,gen_info,Worker.sim_specs,libE_info)            
        else: 
            out = Worker.gen_specs['gen_f'](calc_in,gen_info,Worker.gen_specs,libE_info)
        ### ============================================================================

        assert isinstance(out, tuple), "Calculation output must be a tuple. Worker exiting"
        assert len(out) >= 2, "Calculation output must be at least two elements when a tuple"

        H = out[0]
        gen_info = out[1]
        
        calc_tag = UNSET_TAG #None
        if len(out) >= 3:
            calc_tag = out[2]

        if self.calc_type in self.locations:
            os.chdir(saved_dir)

        #data_out = {'calc_out':H, 'gen_info':gen_info, 'libE_info': libE_info}
        
        #return data_out, calc_tag
        return H, gen_info, libE_info, calc_tag
        
