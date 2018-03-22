"""
libEnsemble worker class
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
import os, shutil
from message_numbers import *

#Worker concept here - may have just work_units...
#This is developed xhffi for evaluating current code but in serial - NOT eventual code
#So will be no. workers fixed and work assigend etc... xtrme*** HH thshit YY||

#May add object attributes for persistent/sim/gen/waiting etc so thats held in object.

class Worker():

    #Class attributes    
    sim_specs = {}
    gen_specs = {}
    
    #Class methods
    
    #Do they all have same sim/gen specs?
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
    
    #If go to multi jobs per worker some attributes will change - or go to job attributes.
    def __init__(self, workerID, H):
        self.dtypes = {}
        self.locations = {}
        self.worker_dir = ""
        self.workerID = workerID
        self.data = {}
        self.calc_type = None
        self.calc_status = None 
        self.isdone = False 
        
        #self.sim_specs = Worker.sim_specs
        #self.gen_specs = Worker.gen_specs
        #self.libE_info = {}
        #self.gen_info = {}
        #self.tag = ""
        #self. = 
        #check whether all need to be attributes - are some just temps....???
        #Check through those set elsewhere and initialise here - #todo!

        self.dtypes[EVAL_SIM_TAG] = H[Worker.sim_specs['in']].dtype
        self.dtypes[EVAL_GEN_TAG] = H[Worker.gen_specs['in']].dtype
       
        if 'sim_dir' in Worker.sim_specs:
            #worker_dir = Worker.sim_specs['sim_dir'] + '_' + str(comm_color) + "_" + str(rank) 
            self.worker_dir = Worker.sim_specs['sim_dir'] + '_' + str(self.workerID)

            if 'sim_dir_prefix' in Worker.sim_specs:
                self.worker_dir =  os.path.join(os.path.expanduser(Worker.sim_specs['sim_dir_prefix']), os.path.split(os.path.abspath(os.path.expanduser(self.worker_dir)))[1])

            assert ~os.path.isdir(self.worker_dir), "Worker directory already exists."
            # if not os.path.exists(worker_dir):
            shutil.copytree(Worker.sim_specs['sim_dir'], self.worker_dir)
            self.locations[EVAL_SIM_TAG] = self.worker_dir 


    #Kind of inclined for the stuff sent for calc_in to be included in Work object - but
    #maybe a reason why not???
    
    def run(self, Work, calc_in):
 
        #Original code - receives libE_info - and STOP_TAG?
        #libE_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        #calc_tag = status.Get_tag()
        #if calc_tag == STOP_TAG: break 
        
        #Think only need self - if going to store between calls to run...
        #Work is always sent currently so those dont need to be stored right!!!
    
        #Could keep all this inside the Work dictionary .....
        #Does it need to be self - what if a just a working variable....
        libE_info = Work['libE_info']
        
        self.calc_type = Work['tag']
        
        #This will be a separate routine - telling worker to kill its job/jobs
        #if self.calc_tag == STOP_TAG: 
        #    self.clean()
        #    return #Any vals to return??? Or call a finalise function???? Clean up...
                 
        #Does it need to be self - what if a just a working variable....
        gen_info = Work['gen_info']
        
        #import pdb; pdb.set_trace()
        #default....
        if calc_in is None:
            calc_in = np.zeros(len(libE_info['H_rows']),dtype=self.dtypes[self.calc_type])
        else:
            calc_in =  calc_in
        
        assert self.calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], "calc_type must either be EVAL_SIM_TAG or EVAL_GEN_TAG"
        
        data_out, tag_out = self._perform_calc(calc_in, gen_info, libE_info) 
        #Note: Removed comm argument
        
        self.calc_status = tag_out
        self.data = data_out
        self.isdone = True
        
        #This will probably be replaced
        if tag_out == STOP_TAG: 
            self.clean()
            return #Any vals to return??? Or call a finalise function???? Clean up...
        
        return
        #return data_out #May want tag_out also
    

    def clean(self):
        # Clean up
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
            os.chdir(self.locations[self.calc_type])

        #Need to check how this works - maybe for sub-comm???
        #if 'persistent' in libE_info and libE_info['persistent']:
        #    libE_info['comm'] = comm
        
        ### ============================== Run calc =======================================
        #import pdb; pdb.set_trace()
        if self.calc_type == EVAL_SIM_TAG:
            out = Worker.sim_specs['sim_f'][0](calc_in,gen_info,Worker.sim_specs,libE_info)
        else: 
            out = Worker.gen_specs['gen_f'](calc_in,gen_info,Worker.gen_specs,libE_info)
        ### ===============================================================================

        assert isinstance(out, tuple), "Calculation output must be a tuple. Worker exiting"
        assert len(out) >= 2, "Calculation output must be at least two elements when a tuple"

        H = out[0]
        gen_info = out[1]
        
        calc_tag = None
        if len(out) >= 3:
            calc_tag = out[2]

        if self.calc_type in self.locations:
            os.chdir(saved_dir)

        #Prob internal only -so _perform_calc ????
        #if 'persistent' in libE_info and libE_info['persistent']:
        #    del libE_info['comm']

        data_out = {'calc_out':H, 'gen_info':gen_info, 'libE_info': libE_info}

        return data_out, calc_tag        
        
