"""
libEnsemble worker class
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
import os, shutil
import socket
from libensemble.message_numbers import *
from libensemble.job_class import Job
import threading
import logging
from libensemble.controller import JobController

#logging.basicConfig(level=logging.DEBUG,
                    #format='(%(threadName)-10s) %(message)s',
                    #)

# All routines in Worker Class have no MPI and can be called regardless of worker
# concurrency mode. The routine worker_main is only used for MPI mode.

#Question over whether have separate MPI_worker module - so no MPI in this module - but for
#now put in here and only load MPI in this routine - NO MPI in Worker Class (below)
#Might change this function name to worker_MPI or something....
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
    
    status = MPI.Status()
    
    Worker.init_workers(sim_specs, gen_specs) # Store in Worker Class
    
    sim_type = comm.recv(buf=None, source=0)
    gen_type = comm.recv(buf=None, source=0)
    
    dtypes = {}
    dtypes[EVAL_SIM_TAG] = sim_type
    dtypes[EVAL_GEN_TAG] = gen_type
    
    #workerID could be MPI rank - or could just be zero (or enumerate if ever have multi workers on an MPI task)
    #workerID = 0
    #print('rank', rank)
    workerID = rank # To use mirror list
    #worker_list.append(new_worker) #For now no worker list as only going to have one - but could easily add.
    
    #worker = Worker(workerID, sim_type, gen_type)
    worker = Worker(workerID)
    #comm, status, dtypes, locations = initialize_worker(c, sim_specs, gen_specs)
    
    print('Worker %d initiated on MPI rank %d on node %s' % (workerID, rank, socket.gethostname()))
    
    while True:
        
        #Note: This comm is experimental change - solution req. discussion.... not there yet!
        #May want a finish to override a kill though came after ... anyway prob.
        #Prob should put in function wait_on_manager()
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)          
        mtag = status.Get_tag()
        if mtag == STOP_TAG:
            man_signal = comm.recv(source=0, tag=STOP_TAG, status=status)
            if man_signal == MAN_SIGNAL_FINISH: #shutdown the worker
                break
            #What if get kill here rather than finish??? (kill means just kill running jobs - dont shut down worker)
            #Would mean not received the message yet for job I have to kill......
            #May then need to deal with message count as could have two to deal with - whereas finish - it doesnt matter.
        else:
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
       
        #Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        #Can add an mpi_iprobe or mpi_test for stop signal
        #calc_tag = status.Get_tag()
        #if calc_tag == STOP_TAG:
            ##if 'clean_jobs' in sim_specs: worker.clean()
            #break

        ##Now repeating stuff in worker.run - maybe
        ##that should just be sent extracted components and comm.recv can either
        ##receive components or receive Work and extract here.        
        libE_info = Work['libE_info']
        calc_type = Work['tag'] #If send components - send tag separately (dont use MPI.status!)
        
        calc_in = np.zeros(len(libE_info['H_rows']),dtype=dtypes[calc_type])
        if len(calc_in) > 0: 
            calc_in = comm.recv(buf=None, source=0)

        ##Either send components or just send Work - for now I'm just sending work - discuss...        
        worker.run(Work, calc_in) #Change to send extracted components required....
        
        ## Receive libE_info from manager and check if STOP_TAG. 
        #libE_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        #calc_tag = status.Get_tag()
        #if calc_tag == STOP_TAG: break
        #gen_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        #calc_in = np.zeros(len(libE_info['H_rows']),dtype=dtypes[calc_tag])
        
        #if len(calc_in) > 0: 
            #calc_in = comm.recv(buf=None, source=0)
            ## for i in calc_in.dtype.names: 
            ##     # d = comm.recv(buf=None, source=0)
            ##     # data = np.empty(calc_in[i].shape, dtype=d)
            ##     data = np.empty(calc_in[i].shape, dtype=calc_in[i].dtype)
            ##     comm.Recv(data,source=0)
            ##     calc_in[i] = data 
        
        #End signal from inside worker....
        #Do I want to send data back first (if got it??) - and do I want to send the kill info back to manager?
        #if worker.calc_status == STOP_TAG:
        if worker.calc_status == MAN_SIGNAL_FINISH:
            break
        

        #print('tag',worker.calc_status)
        #comm.send(obj=worker, dest=0, tag=worker.calc_status)      # Whole object (inc. joblist if set up for example - so can print timings)
        comm.send(obj=worker, dest=0) #blocking
        #comm.isend(obj=worker, dest=0) # Non-blocking - failing but if to recieve a kill when blocking- manger has to recv this first
        #comm.send(obj=worker.data, dest=0, tag=tag_out) # Just worker.data - as was doing before 
        
        #May be that on getting a STOP signal from manager - return the whole worker - so can get jobs info - but
        #not every time - rest time just return data.
    
    if 'clean_jobs' in sim_specs:
        worker.clean()
    
    # Print joblist here
    timing_file = 'timing.dat.w' + str(worker.workerID)
    with open(timing_file,'w') as f:
        f.write("Worker %d:\n" % (worker.workerID))
        #for j, jb in enumerate(worker.joblist):
        for jb in worker.joblist:            
            jb.printjob(f)
    
    #Ideally send message to manager to say when done - before manager collates files.
    #comm.send(obj=None, dest=0, tag = WORKER_DONE) #do it jeffs way!
    #comm.isend(obj=WORKER_DONE, dest=0, tag = WORKER_DONE) #do it jeffs way!
    
    #print('Sent it back worker')
    
    #Destroy worker object???

# NO MPI in here
class Worker():

    #Class attributes    
    sim_specs = {}
    gen_specs = {}
    
    #Class methods
    
    #Do they all have same sim/gen specs?
    #Could also do the dtypes at this level.... (and I dont think really need to provide.....)
    def init_workers(sim_specs_in, gen_specs_in):
        
        #Class attributes? Maybe should be worker specific??
        Worker.sim_specs = sim_specs_in
        Worker.gen_specs = gen_specs_in
        #May setup sim_dir_prefix here
        
        
        #always same? - May be various things always the same ????
        #dtypes = {}
        #dtypes[EVAL_SIM_TAG] = H[sim_specs['in']].dtype
        #dtypes[EVAL_GEN_TAG] = H[gen_specs['in']].dtype
        
    def get_worker(worker_list,workerID):
        #add some error handling...
        for worker in worker_list:
            if worker.workerID == workerID:
                return worker
        #Does not exist
        return None

    def get_worker_index(worker_list,workerID):
        #add some error handling...
        #import pdb;pdb.set_trace()
        index=0
        for worker in worker_list:
            if worker.workerID == workerID:
                return index
            index += 1
        #Does not exist
        return None
    
    #If go to multi jobs per worker some attributes will change - or go to job attributes.
    #def __init__(self, workerID, H):
    def __init__(self, workerID, empty=False):
    #def __init__(self, workerID, sim_type, gen_type):        
        #self.dtypes = {}
        self.locations = {}
        self.worker_dir = ""
        self.workerID = workerID
        self.data = {}
        self.calc_type = None
        self.calc_status = UNSET_TAG #From message_numbers
        self.isdone = False
        self.joblist = []
        
        #self.sim_specs = Worker.sim_specs
        #self.gen_specs = Worker.gen_specs       
        
        if not empty:
            if 'sim_dir' in Worker.sim_specs:
                #worker_dir = Worker.sim_specs['sim_dir'] + '_' + str(comm_color) + "_" + str(rank) 
                self.worker_dir = Worker.sim_specs['sim_dir'] + '_' + str(self.workerID)
    
                if 'sim_dir_prefix' in Worker.sim_specs:
                    self.worker_dir =  os.path.join(os.path.expanduser(Worker.sim_specs['sim_dir_prefix']), os.path.split(os.path.abspath(os.path.expanduser(self.worker_dir)))[1])
    
                assert ~os.path.isdir(self.worker_dir), "Worker directory already exists."
                # if not os.path.exists(worker_dir):
                shutil.copytree(Worker.sim_specs['sim_dir'], self.worker_dir)
                self.locations[EVAL_SIM_TAG] = self.worker_dir #May change to self.sim_dir and self.gen_dir
                
                #Optional - set workerID in job_controller - so will be added to jobnames
                jobctl = JobController.controller
                jobctl.set_workerID(workerID)


    #worker.run
    def run(self, Work, calc_in):
        
        self.data = {}
        self.calc_type = None
        self.calc_status = UNSET_TAG #From message_numbers
        self.isdone = False  
        
        #t = threading.currentThread()
        #logging.debug('Running thread %s on worker %d', t.getName(), self.workerID)
        
        #Add a job (This is user job - currently different level to system job (in JobController). Discuss. 
        #User job object is to contain info to be stored for each job (in joblist)
        job = Job()
        self.joblist.append(job)
        
        #Timing will include setup/teardown
        job.start_timer()
        
        #Could keep all this inside the Work dictionary if sending all Work ...
        libE_info = Work['libE_info']
        self.calc_type = Work['tag']
        job.calc_type = Work['tag']
        gen_info = Work['gen_info']        
        #logging.debug('Running thread %s on worker %d %s', t.getName(), self.workerID, self.calc_type)
        
        assert self.calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"
        
        data_out, tag_out = self._perform_calc(calc_in, gen_info, libE_info)
        self.calc_status = tag_out
        
        #This is a libe feature that is to be reviewed for best solution
        if self.calc_status == MAN_SIGNAL_FINISH:   #Think these should only be used for message tags?
            job.status = "Manager killed on finish" #Currently a string/description
        elif self.calc_status == MAN_SIGNAL_KILL: 
            job.status = "Manager killed job"
        elif self.calc_status == WORKER_KILL:
            job.status = "Worker killed job"
        elif self.calc_status == JOB_FAILED:
            job.status = "Job Failed"        
        else:
            job.status = "Completed"
            
        self.data = data_out
        self.isdone = True
        
        job.stop_timer()
                
        return # Can retrieve output from worker.data
    
    # Do we want to be removing these dirs by default??? Maybe an option
    # Could be option in sim_specs - "clean_jobdirs"
    def clean(self):
        # Clean up - may need to chdir to saved dir also (saved_dir cld be object attribute)
        for loc in self.locations.values():
            shutil.rmtree(loc)
        return
    
    def isdone(self):
        #Poll job - Dont need function
        return self.isdone

    #Prob internal only
    def _perform_calc(self, calc_in, gen_info, libE_info):
        if self.calc_type in self.locations:
            saved_dir = os.getcwd()
            #print('current dir in _perform_calc is ', saved_dir)
            #logging.debug('current dir in _perform_calc is  %s', saved_dir)
            os.chdir(self.locations[self.calc_type])

        #Need to check how this works
        #if 'persistent' in libE_info and libE_info['persistent']:
        #    libE_info['comm'] = comm
        
        ### ============================== Run calc =======================================
        #import pdb; pdb.set_trace()
        if self.calc_type == EVAL_SIM_TAG:
            #out = Worker.sim_specs['sim_f'][0](calc_in,gen_info,Worker.sim_specs,libE_info)
            
            #experiment - cld pass workerID OR worker (ie. pass "self") OR pass the job - with workerID contained.
            #Also alternative route through registry/job_controller
            out = Worker.sim_specs['sim_f'][0](calc_in,gen_info,Worker.sim_specs,libE_info, self.workerID)
            #out = Worker.sim_specs['sim_f'][0](calc_in,gen_info,Worker.sim_specs,libE_info)            
        else: 
            out = Worker.gen_specs['gen_f'](calc_in,gen_info,Worker.gen_specs,libE_info)
        ### ===============================================================================

        assert isinstance(out, tuple), "Calculation output must be a tuple. Worker exiting"
        assert len(out) >= 2, "Calculation output must be at least two elements when a tuple"

        H = out[0]
        gen_info = out[1]
        
        calc_tag = UNSET_TAG #None
        if len(out) >= 3:
            calc_tag = out[2]

        if self.calc_type in self.locations:
            os.chdir(saved_dir)

        #if 'persistent' in libE_info and libE_info['persistent']:
        #    del libE_info['comm']

        data_out = {'calc_out':H, 'gen_info':gen_info, 'libE_info': libE_info}

        return data_out, calc_tag
        
