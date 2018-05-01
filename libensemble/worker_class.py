"""
libEnsemble worker class
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
import os, shutil
from libensemble.message_numbers import *
from libensemble.job_class import Job
import threading
import logging
#import pdb

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

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
    print('rank', rank)
    workerID = rank # To use mirror list
    #worker_list.append(new_worker) #For now no worker list as only going to have one - but could easily add.
    
    #worker = Worker(workerID, sim_type, gen_type)
    worker = Worker(workerID)
    #comm, status, dtypes, locations = initialize_worker(c, sim_specs, gen_specs)
    
    while True:
        
        #Hmmm - I want to change this - now repeating stuff in worker.run - maybe
        #that should just be sent extracted components and comm.recv can either
        #receive components or receive Work and extract here.
        
        #Either send components or just send Work - for now I'm just sending work - discuss...
        Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        
        #This is for STOP only - and I think it should be a separate MPI - not using status!
        #Can add an mpi_iprobe or mpi_test for stop signal
        calc_tag = status.Get_tag()
        if calc_tag == STOP_TAG:
            #if 'clean_jobs' in sim_specs: worker.clean()
            break
        
        libE_info = Work['libE_info']
        calc_type = Work['tag'] #If send components - send tag separately (dont use MPI.status!)
        
        calc_in = np.zeros(len(libE_info['H_rows']),dtype=dtypes[calc_type])
        if len(calc_in) > 0: 
            calc_in = comm.recv(buf=None, source=0)
        
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
        if worker.calc_status == STOP_TAG:
            #if 'clean_jobs' in sim_specs: worker.clean()
            break
        

        #Why did I call data rather than data_out ???? - change back to data_out
        #Also shall I send back entire worker object - is that unnecessary??? Is it too much - esp if inc. all job data???
        #to do - remove tag right ...

        #print('tag',worker.calc_status)
        #comm.send(obj=worker, dest=0, tag=worker.calc_status)      # Whole object (inc. joblist if set up for example - so can print timings)
        comm.send(obj=worker, dest=0)
        #comm.send(obj=worker.data, dest=0, tag=tag_out) # Just worker.data - as was doing before 
        
        #Another idea might be - that on getting a STOP signal from manager - return the whole worker - so can get jobs info - but
        #not every time - rest time just return data. This would mean a less overlap in manager receive routines though - as the
        #serial case is based around a worker object.
    
    if 'clean_jobs' in sim_specs: worker.clean()
    
    # Print joblist here
    timing_file = 'timing.dat.w' + str(worker.workerID)
    with open(timing_file,'w') as f:
        f.write("Worker %d:\n" % (worker.workerID))
        for j, jb in enumerate(worker.joblist):
            f.write("   Job %d: %s Tot: %f\n" % (j,jb.get_type(),jb.time))
            
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
        self.isdone = False #Shld this be per job?
        self.joblist = []
        
        #self.sim_specs = Worker.sim_specs
        #self.gen_specs = Worker.gen_specs
        #self.libE_info = {}
        #self.gen_info = {}
        #self.tag = ""
        #self. = 
        #check whether all need to be attributes - are some just temps....???
        #Check through those set elsewhere and initialise here - #todo!

        #self.dtypes[EVAL_SIM_TAG] = H[Worker.sim_specs['in']].dtype
        #self.dtypes[EVAL_GEN_TAG] = H[Worker.gen_specs['in']].dtype
        
        #self.dtypes[EVAL_SIM_TAG] = sim_type
        #self.dtypes[EVAL_GEN_TAG] = gen_type
        
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
    

    #Kind of inclined for the stuff sent for calc_in to be included in Work object - but
    #maybe a reason why not???
    #worker.run - maybe bring _perform_calc into this - as MPI uses wrapper code anyway
    def run(self, Work, calc_in):
 
        #Original code - receives libE_info - and STOP_TAG?
        #libE_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        #calc_tag = status.Get_tag()
        #if calc_tag == STOP_TAG: break 
        
        #Think only need self - if going to store between calls to run...
        #Work is always sent currently so those dont need to be stored right!!!
        
        #pdb.set_trace()
        t = threading.currentThread()
        #logging.debug('Running thread %s on worker %d', t.getName(), self.workerID)
        
        #Add a job - decide whether or not to combine with Work - which shld also be an object - or is work unit and job different.
        #ie. job contains info on resource -eg. process_id - output like run-time etc... where as a work_unit can run anywhere - its work to be done.
        
        job = Job()
        self.joblist.append(job)
        
        #For now timing will include setup/teardown
        job.start_timer()
        
    
        #Could keep all this inside the Work dictionary .....
        #Does it need to be self - what if a just a working variable....
        libE_info = Work['libE_info']
        
        self.calc_type = Work['tag']
        #logging.debug('Running thread %s on worker %d %s', t.getName(), self.workerID,self.calc_type)
        
        #maybe shld just a job attribute - and use that - but for now just setting job to have a record of jobs at end.
        job.calc_type = Work['tag']
        
        #logging.debug('Running thread %s on worker %d %s', t.getName(), self.workerID, self.calc_type)
        logging.debug('Running thread %s on worker %d %s', t.getName(), self.workerID, job.get_type())
        #import pdb;pdb.set_trace()
        
        #This will be a separate routine - telling worker to kill its job/jobs
        #if self.calc_tag == STOP_TAG: 
        #    self.clean()
        #    return #Any vals to return??? Or call a finalise function???? Clean up...
                 
        #Does it need to be self - what if a just a working variable....
        gen_info = Work['gen_info']
        
        #import pdb; pdb.set_trace()
        #default....
        #if calc_in is None:
            #calc_in = np.zeros(len(libE_info['H_rows']),dtype=self.dtypes[self.calc_type])
        #else:
            #calc_in =  calc_in
        
        assert self.calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"
        
        data_out, tag_out = self._perform_calc(calc_in, gen_info, libE_info) 
        #Note: Removed comm argument
        
        self.calc_status = tag_out
        self.data = data_out
        self.isdone = True
        
        job.stop_timer()
        
        #Dont think this makes sense here - put in worker loop in MPI case
        ##End signal from inside worker....
        #if tag_out == STOP_TAG: 
            #if 'clean_jobs' in sim_specs: self.clean()
                
        return # Can retrieve output from worker.data
    
        #return data_out #May want tag_out also
    
    # Do we want to be removing these by default??? Maybe an option
    # Could be option in sim_specs or something - clean up job dirs...
    def clean(self):
        # Clean up - may need to chdir to saved dir also (saved_dir cld be object attribute)
        for loc in self.locations.values():
            shutil.rmtree(loc)
        return
    
    def isdone(self):
        #Poll job - for now its blocking so return self.isdone.
        return self.isdone
    
    #Dont think need getter funcs in python - can access atrributes externally but feels weird!
    #def calc_type(self):
    #    return self.calc_type


    #Prob internal only -so _perform_calc ????
    def _perform_calc(self, calc_in, gen_info, libE_info):
        if self.calc_type in self.locations:
            saved_dir = os.getcwd()
            #print('current dir in _perform_calc is ', saved_dir)
            logging.debug('current dir in _perform_calc is  %s', saved_dir)
            os.chdir(self.locations[self.calc_type])

        #Need to check how this works - maybe for sub-comm???
        #if 'persistent' in libE_info and libE_info['persistent']:
        #    libE_info['comm'] = comm
        
        ### ============================== Run calc =======================================
        #import pdb; pdb.set_trace()
        if self.calc_type == EVAL_SIM_TAG:
            #out = Worker.sim_specs['sim_f'][0](calc_in,gen_info,Worker.sim_specs,libE_info)
            
            #experiment - cld pass workerID or cld pass worker (if import worker user-side) - ie. pass "self"
            #           - and then cld access anything here - will need to pass job of course
            #           - maybe thats all I need to pass - and job could contain workerID !!!!
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

        #Prob internal only -so _perform_calc ????
        #if 'persistent' in libE_info and libE_info['persistent']:
        #    del libE_info['comm']

        data_out = {'calc_out':H, 'gen_info':gen_info, 'libE_info': libE_info}

        return data_out, calc_tag
        
